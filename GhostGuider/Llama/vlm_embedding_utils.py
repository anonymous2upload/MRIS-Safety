

import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
import logging

from token_utils import get_last_valid_token_index

logger = logging.getLogger(__name__)


class Llama32VisionEmbeddingExtractor:
    
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        
        self.use_device_map = hasattr(self.model, 'hf_device_map')
        
        if not self.use_device_map:
            self.model.to(device)
        
        self.model.eval()
        self.num_layers = self._get_num_layers()
    
    def _get_num_layers(self) -> int:
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
                return config.text_config.num_hidden_layers
            
            if hasattr(config, 'num_hidden_layers'):
                return config.num_hidden_layers
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        
        logger.warning("Unable to determine model layer count, using default value of 32")
        return 32
    
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
    
    def _build_llama_prompt(self, query: str, mode: str = "image") -> str:
        if mode == "image":
            formatted_prompt = f"<|image|><|begin_of_text|>{query}"
        else:
            formatted_prompt = f"<|begin_of_text|>{query}"
        
        return formatted_prompt
    
    def _prepare_inputs(self, image: Optional[Image.Image], query: str) -> Dict:
        if image is not None:
            prompt = self._build_llama_prompt(query, mode="image")
            inputs = self.processor(
                image,
                prompt,
                return_tensors="pt"
            )
        else:
            prompt = self._build_llama_prompt(query, mode="text")
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            )
        
        if self.use_device_map:
            input_device = self.get_input_device()
            inputs = {k: v.to(input_device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        return inputs
    
    def _get_last_valid_token_indices(
        self, 
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        batch_size: int,
        device: torch.device
    ) -> torch.LongTensor:
        return get_last_valid_token_index(
            attention_mask=attention_mask,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device
        )
    
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
        
        last_valid_indices = self._get_last_valid_token_indices(
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
        
        logger.info(f"Starting to extract embeddings for {len(data)} samples...")
        
        for i, item in enumerate(data):
            if i % batch_size == 0:
                logger.info(f"Processing progress: {i}/{len(data)} ({i*100//len(data)}%)")
            
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


def create_llama_extractor(
    model_path: str,
    device: str = 'cuda',
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    **kwargs
) -> Llama32VisionEmbeddingExtractor:
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    
    logger.info(f"Creating Llama-3.2-Vision Embedding Extractor: {model_path}")
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model_kwargs = {
        'torch_dtype': torch_dtype,
        'device_map': 'auto',
        **kwargs
    }
    
    logger.info("Loading Llama-3.2-Vision model...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )
    logger.info("Model loaded successfully")
    
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info("Processor loaded successfully")
    
    extractor = Llama32VisionEmbeddingExtractor(
        model=model,
        processor=processor,
        device=device,
    )
    
    logger.info(f"Embedding extractor created, model layers: {extractor.num_layers}")
    
    return extractor


QwenVLEmbeddingExtractor = Llama32VisionEmbeddingExtractor
create_qwen_extractor = create_llama_extractor
