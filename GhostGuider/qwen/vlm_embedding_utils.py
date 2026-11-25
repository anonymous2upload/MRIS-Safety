import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
import logging

from token_utils import get_last_valid_token_index

logger = logging.getLogger(__name__)


class QwenVLEmbeddingExtractor:
    
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        
        self.use_device_map = hasattr(self.model, 'hf_device_map')
        
        if not self.use_device_map:
            self.model.to(device)
        
        self.model.eval()
        self.num_layers = self._get_num_layers()
        
        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        except ImportError:
            logger.warning("qwen_vl_utils not installed, using basic functionality")
            self.process_vision_info = None
    
    def _get_num_layers(self) -> int:
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        else:
            raise ValueError(f"Cannot determine Qwen-VL model layer count")
    
    def get_input_device(self):
        if self.use_device_map:
            try:
                first_param = next(self.model.parameters())
                return first_param.device
            except StopIteration:
                return torch.device(self.device)
        else:
            return torch.device(self.device)
    
    def load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert('RGB')
    
    def _prepare_inputs(self, image: Image.Image, query: str) -> Dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if self.process_vision_info is not None:
            image_inputs, video_inputs = self.process_vision_info(messages)
        else:
            image_inputs = [image]
            video_inputs = None
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        if self.use_device_map:
            input_device = self.get_input_device()
            inputs = {k: v.to(input_device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        return inputs
    
    def extract_hidden_states(
        self, 
        image_path: str,
        query: str,
        layers: Optional[List[int]] = None
    ) -> np.ndarray:
        image = self.load_image(image_path)
        inputs = self._prepare_inputs(image, query)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        hidden_states = outputs.hidden_states
        attention_mask = inputs.get('attention_mask', None)
        
        num_available_layers = len(hidden_states) - 1
        
        if layers is None:
            layers = list(range(num_available_layers))
        
        batch_size = hidden_states[0].shape[0]
        seq_len = hidden_states[0].shape[1]
        device = hidden_states[0].device

        last_valid_indices = get_last_valid_token_index(
            attention_mask=attention_mask,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device
        )
        
        last_valid_idx = last_valid_indices[0].item()
        
        layer_activations = []
        for layer_idx in layers:
            hidden_state = hidden_states[layer_idx + 1]
            last_token_activation = hidden_state[0, last_valid_idx, :]
            layer_activations.append(last_token_activation.cpu().float().numpy())
        
        result = np.array(layer_activations)
        
        return result
    
    def extract_batch_embeddings(
        self,
        data: List[Dict],
        layers: Optional[List[int]] = None,
        batch_size: int = 1
    ) -> np.ndarray:
        all_embeddings = []
        
        logger.info(f"Starting extraction of {len(data)} sample embeddings...")
        
        for i, item in enumerate(data):
            if i % batch_size == 0:
                logger.info(f"Progress: {i}/{len(data)} ({i*100//len(data)}%)")
            
            try:
                embedding = self.extract_hidden_states(
                    item['image_path'],
                    item['query'],
                    layers
                )
                all_embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Failed to extract embedding [{item.get('image_path', 'unknown')}]: {e}")
                continue
        
        result = np.array(all_embeddings)
        logger.info(f"Extraction complete! Final shape: {result.shape}")
        
        return result


def create_qwen_extractor(
    model_path: str,
    device: str = 'cuda',
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    attn_implementation: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    **kwargs
) -> QwenVLEmbeddingExtractor:
    from transformers import AutoProcessor, AutoConfig
    
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = config.model_type
        
        if model_type == 'qwen2_5_vl':
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
            logger.info("Detected Qwen2.5-VL model")
        elif model_type == 'qwen2_vl':
            from transformers import Qwen2VLForConditionalGeneration as ModelClass
            logger.info("Detected Qwen2-VL model")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    except Exception as e:
        logger.error(f"Cannot determine model type: {e}")
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        logger.warning("Using default Qwen2_5_VLForConditionalGeneration")
    
    logger.info(f"Creating Qwen-VL Embedding Extractor: {model_path}")
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model_kwargs = {
        'torch_dtype': torch_dtype,
        'device_map': 'auto',
        'trust_remote_code': trust_remote_code,
        **kwargs
    }
    
    if attn_implementation is not None:
        model_kwargs['attn_implementation'] = attn_implementation
    
    logger.info("Loading model...")
    model = ModelClass.from_pretrained(
        model_path,
        **model_kwargs
    )
    logger.info("Model loaded successfully")
    
    processor_kwargs = {'trust_remote_code': trust_remote_code}
    if min_pixels is not None and max_pixels is not None:
        processor_kwargs['min_pixels'] = min_pixels
        processor_kwargs['max_pixels'] = max_pixels
    
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    logger.info("Processor loaded successfully")
    
    extractor = QwenVLEmbeddingExtractor(
        model=model,
        processor=processor,
        device=device,
    )
    
    logger.info("Embedding extractor created successfully")
    
    return extractor
