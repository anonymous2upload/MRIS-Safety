
import torch
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import os
from typing import List, Dict, Optional, Tuple

from vlm_embedding_utils import create_llama_extractor
from nullspace_utils import (
    null_space_projection_l,
    cal_tilde_delta_with_regularization_l,
    cal_steering_matrix_l,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_json_data(data_path: str, sample_ratio: Optional[float] = None) -> List[Dict]:
    logger.info(f"Loading data: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    total_size = len(data)
    
    if sample_ratio is not None:
        if not 0 < sample_ratio <= 1:
            raise ValueError(f"sample_ratio must be in range (0, 1], current value: {sample_ratio}")
        
        sample_size = int(total_size * sample_ratio)
        if sample_size < total_size:
            import random
            data = random.sample(data, sample_size)
            logger.info(f"Sampling {sample_ratio*100:.1f}% data: {sample_size}/{total_size} samples")
        else:
            logger.info(f"Using all {total_size} samples")
    else:
        logger.info(f"Using all {total_size} samples")
    
    return data


def load_mris_data(mris_path: str, sample_ratio: Optional[float] = None) -> List[Dict]:
    logger.info(f"Loading MRIS data: {mris_path}")
    with open(mris_path, 'r') as f:
        data = json.load(f)
    
    malicious_data = []
    for item in data:
        predicted_type = item.get('predicted_type', '')
        if predicted_type in ['Ignore_Answer', 'Ignore_Refuse']:
            malicious_data.append({
                'image_path': item['mris_image_path'],
                'query': item['rewrite_question'],
                'type': predicted_type
            })
    
    total_size = len(malicious_data)
    logger.info(f"Found {total_size} MRIS attack samples")
    
    if sample_ratio is not None:
        if not 0 < sample_ratio <= 1:
            raise ValueError(f"sample_ratio must be in range (0, 1], current value: {sample_ratio}")
        
        sample_size = int(total_size * sample_ratio)
        if sample_size < total_size:
            import random
            malicious_data = random.sample(malicious_data, sample_size)
            logger.info(f"Sampling {sample_ratio*100:.1f}% data: {sample_size}/{total_size} samples")
    
    return malicious_data


def load_risk_vectors(risk_vector_path: str, device: str = 'cuda') -> torch.Tensor:
    logger.info(f"Loading risk identification vectors: {risk_vector_path}")
    
    if risk_vector_path.endswith('.npy'):
        risk_vectors = np.load(risk_vector_path)
    elif risk_vector_path.endswith('.json'):
        with open(risk_vector_path, 'r') as f:
            data = json.load(f)
            risk_vectors = np.array(data['risk_vectors'])
    else:
        raise ValueError("Risk vector file must be in .npy or .json format")
    
    logger.info(f"Risk vector shape: {risk_vectors.shape}")
    
    return torch.from_numpy(risk_vectors).to(device).float()


def parse_layer_config(config_str: str) -> List[Tuple[int, float]]:
    if not config_str:
        return None
    
    layers_ratio_list = []
    
    for item in config_str.split(','):
        item = item.strip()
        if ':' in item:
            layer_part, ratio = item.split(':')
            ratio = float(ratio)
            
            if '-' in layer_part:
                start, end = map(int, layer_part.split('-'))
                for layer in range(start, end + 1):
                    layers_ratio_list.append((layer, ratio))
            else:
                layer = int(layer_part)
                layers_ratio_list.append((layer, ratio))
        else:
            raise ValueError(f"Invalid configuration item: {item}")
    
    return layers_ratio_list


def compute_llama_dual_protected_steering_matrices(
    model_path: str,
    benign_data_path: str,
    harmful_data_path: str,
    mris_data_path: str,
    risk_vector_path: str,
    output_path: str,
    layers_config: Optional[str] = None,
    default_nullspace_ratio: float = 0.6,
    lambda_reg: float = 10.0,
    benign_sample_ratio: Optional[float] = None,
    harmful_sample_ratio: Optional[float] = None,
    malicious_sample_ratio: Optional[float] = None,
    batch_size: int = 4,
    device: str = 'cuda',
):
    
    logger.info("="*70)
    logger.info("Creating Llama-3.2-Vision Embedding Extractor")
    logger.info("="*70)
    
    extractor = create_llama_extractor(model_path, device)
    
    logger.info("\n" + "="*70)
    logger.info("Loading Data")
    logger.info("="*70)
    benign_data = load_json_data(benign_data_path, benign_sample_ratio)
    harmful_data = load_json_data(harmful_data_path, harmful_sample_ratio)
    malicious_data = load_mris_data(mris_data_path, malicious_sample_ratio)
    risk_vectors = load_risk_vectors(risk_vector_path, device)
    
    logger.info("\n" + "="*70)
    logger.info("Extracting Benign Embeddings")
    logger.info("="*70)
    benign_embeddings = extractor.extract_batch_embeddings(
        benign_data, layers=None, batch_size=batch_size
    )
    
    logger.info("\n" + "="*70)
    logger.info("Extracting Harmful Embeddings (correctly rejected)")
    logger.info("="*70)
    harmful_embeddings = extractor.extract_batch_embeddings(
        harmful_data, layers=None, batch_size=batch_size
    )
    
    logger.info("\n" + "="*70)
    logger.info("Extracting MRIS Malicious Embeddings")
    logger.info("="*70)
    malicious_embeddings = extractor.extract_batch_embeddings(
        malicious_data, layers=None, batch_size=batch_size
    )
    
    H_benign = torch.from_numpy(benign_embeddings).to(device).float()
    H_harmful = torch.from_numpy(harmful_embeddings).to(device).float()
    H_mris = torch.from_numpy(malicious_embeddings).to(device).float()
    
    logger.info(f"\nH_benign shape: {H_benign.shape}")
    logger.info(f"H_harmful shape: {H_harmful.shape}")
    logger.info(f"H_mris shape: {H_mris.shape}")
    
    if layers_config:
        layers_ratio_list = parse_layer_config(layers_config)
        logger.info(f"\nUsing configured layers: {layers_ratio_list}")
    else:
        num_layers = H_benign.shape[1]
        layers_ratio_list = [(i, default_nullspace_ratio) for i in range(num_layers)]
        logger.info(f"\nUsing all {num_layers} layers, nullspace_ratio={default_nullspace_ratio}")
    
    num_layers = risk_vectors.shape[0]
    hidden_dim = risk_vectors.shape[1]
    
    logger.info("\n" + "="*70)
    logger.info("Computing Dual-Protected Steering Matrices")
    logger.info("="*70)
    
    P = torch.zeros(num_layers, hidden_dim, hidden_dim, device=device)
    tilde_delta = torch.zeros(num_layers, hidden_dim, hidden_dim, device=device)
    steering_matrix = torch.zeros(num_layers, hidden_dim, hidden_dim, device=device)
    
    for layer, ratio in tqdm(layers_ratio_list, desc="Computing steering matrix for each layer"):
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing layer {layer}, nullspace_ratio: {ratio}")
        logger.info(f"{'='*70}")
        
        H_protected_layer = torch.cat([
            H_benign[:, layer, :],
            H_harmful[:, layer, :]
        ], dim=0)
        
        logger.info(f"Protected activations: benign={H_benign.shape[0]}, "
                   f"harmful={H_harmful.shape[0]}, "
                   f"total={H_protected_layer.shape[0]}")
        
        P_layer = null_space_projection_l(
            H_protected_layer,
            abs_nullspace_ratio=ratio
        )
        P[layer] = P_layer
        
        test_benign = H_benign[:min(5, H_benign.shape[0]), layer, :]
        steering_on_benign = test_benign @ P_layer
        benign_norm = torch.norm(steering_on_benign).item()
        
        test_harmful = H_harmful[:min(5, H_harmful.shape[0]), layer, :]
        steering_on_harmful = test_harmful @ P_layer
        harmful_norm = torch.norm(steering_on_harmful).item()
        
        logger.info(f"Null Space verification:")
        logger.info(f"  ✓ Benign steering norm:  {benign_norm:.6f} (should be close to 0)")
        logger.info(f"  ✓ Harmful steering norm: {harmful_norm:.6f} (should be close to 0)")
        
        tilde_delta_layer = cal_tilde_delta_with_regularization_l(
            H_mris[:, layer, :],
            P_layer,
            risk_vectors[layer],
            lambda_reg=lambda_reg,
            device=device
        )
        tilde_delta[layer] = tilde_delta_layer
        
        steering_matrix_layer = cal_steering_matrix_l(
            P_layer,
            tilde_delta_layer,
            device=device
        )
        steering_matrix[layer] = steering_matrix_layer
        
        test_mris = H_mris[:min(5, H_mris.shape[0]), layer, :]
        steering_on_mris = test_mris @ steering_matrix_layer.T
        risk_vector_expanded = risk_vectors[layer].unsqueeze(0).expand(test_mris.shape[0], -1)
        alignment = torch.nn.functional.cosine_similarity(
            steering_on_mris,
            risk_vector_expanded,
            dim=1
        ).mean().item()
        
        logger.info(f"MRIS Steering effectiveness:")
        logger.info(f"  ✓ Cosine similarity with risk vector: {alignment:.4f} (should be >0.85)")
        logger.info(f"  ✓ Steering matrix norm: {torch.norm(steering_matrix_layer).item():.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("Saving Results")
    logger.info("="*70)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(steering_matrix, output_path)
    logger.info(f"✓ Steering matrix saved: {output_path}")
    
    full_save_path = output_path.replace('.pt', '_full.pt')
    save_dict = {
        'steering_matrices': steering_matrix.cpu(),
        'P_matrices': P.cpu(),
        'Delta_matrices': tilde_delta.cpu(),
        'metadata': {
            'model_type': 'llama-3.2-vision',
            'model_path': model_path,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'num_benign_samples': len(benign_data),
            'num_harmful_samples': len(harmful_data),
            'num_mris_samples': len(malicious_data),
            'benign_sample_ratio': benign_sample_ratio,
            'harmful_sample_ratio': harmful_sample_ratio,
            'malicious_sample_ratio': malicious_sample_ratio,
            'layers_config': layers_config,
            'default_nullspace_ratio': default_nullspace_ratio,
            'lambda_reg': lambda_reg,
            'layers_computed': [layer for layer, _ in layers_ratio_list],
            'method': 'dual_protected_ghostguider_llama32',
        }
    }
    torch.save(save_dict, full_save_path)
    logger.info(f"✓ Full information saved: {full_save_path}")
    
    json_path = output_path.replace('.pt', '_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(save_dict['metadata'], f, indent=2)
    logger.info(f"✓ Metadata saved: {json_path}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Completed!")
    logger.info("="*70)
    
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Dual-Protected Steering Matrices for Llama-3.2-Vision"
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Llama-3.2-Vision model path")
    
    parser.add_argument("--benign_data", type=str, required=True,
                       help="Benign data JSON file path")
    parser.add_argument("--harmful_data", type=str, required=True,
                       help="Harmful data JSON file path (correctly rejected by model)")
    parser.add_argument("--mris_data", type=str, required=True,
                       help="MRIS data JSON file path")
    parser.add_argument("--risk_vectors", type=str, required=True,
                       help="Risk identification vector file path (.npy or .json)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for steering matrices (.pt)")
    
    parser.add_argument("--layers_config", type=str, default="16-32:0.6",
                       help="Layer configuration, format: '10:0.6,15:0.6' or '10-20:0.6'")
    parser.add_argument("--nullspace_ratio", type=float, default=0.6,
                       help="Default null space ratio, default 0.6")
    parser.add_argument("--lambda_reg", type=float, default=10.0,
                       help="Regularization coefficient, default 10.0")
    
    parser.add_argument("--benign_sample_ratio", type=float, default=0.4,
                       help="Benign data sampling ratio, range (0, 1], e.g. 0.8 means sampling 80%% of data")
    parser.add_argument("--harmful_sample_ratio", type=float, default=0.4,
                       help="Harmful data sampling ratio, range (0, 1], e.g. 0.8 means sampling 80%% of data")
    parser.add_argument("--malicious_sample_ratio", type=float, default=0.4,
                       help="MRIS data sampling ratio, range (0, 1], e.g. 0.8 means sampling 80%% of data")
    
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size, default 4")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Computation device, default cuda")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    for ratio_name, ratio_value in [
        ('benign_sample_ratio', args.benign_sample_ratio),
        ('harmful_sample_ratio', args.harmful_sample_ratio),
        ('malicious_sample_ratio', args.malicious_sample_ratio)
    ]:
        if ratio_value is not None and not (0 < ratio_value <= 1):
            raise ValueError(f"{ratio_name} must be in range (0, 1], current value: {ratio_value}")
    
    compute_llama_dual_protected_steering_matrices(
        model_path=args.model_path,
        benign_data_path=args.benign_data,
        harmful_data_path=args.harmful_data,
        mris_data_path=args.mris_data,
        risk_vector_path=args.risk_vectors,
        output_path=args.output_path,
        layers_config=args.layers_config,
        default_nullspace_ratio=args.nullspace_ratio,
        lambda_reg=args.lambda_reg,
        benign_sample_ratio=args.benign_sample_ratio,
        harmful_sample_ratio=args.harmful_sample_ratio,
        malicious_sample_ratio=args.malicious_sample_ratio,
        batch_size=args.batch_size,
        device=args.device,
    )
