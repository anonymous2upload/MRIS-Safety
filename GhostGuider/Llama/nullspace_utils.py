
import torch
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def null_space_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=0.0):
    _, S, Vh = torch.linalg.svd(A.T @ A, full_matrices=True)
    M, N = A.shape[0], A.shape[1]
    
    if abs_nullspace_ratio > 0:
        num = int(N * abs_nullspace_ratio)
    else:    
        S_ = torch.sqrt(S)
        rcond = torch.finfo(S.dtype).eps * max(M, N)
        tol = torch.amax(S_) * rcond
        num = torch.sum(S_ < tol)
    
        if num / N < min_null_space_ratio:
            num = int(N * min_null_space_ratio)
    
    logger.info(f"final null space ratio: {num / N:.4f}")
    Q = Vh[-num:,:].T.conj()
    return Q


def null_space_projection_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=0.0):
    Q = null_space_l(A, min_null_space_ratio, abs_nullspace_ratio)
    P = Q @ Q.T
    return P


def cal_tilde_delta_with_regularization_l(
    H_h_layer, P_layer, refusal_vector, lambda_reg, device="cuda:0"):
    H_h_layer = H_h_layer.to(device)
    P_layer = P_layer.to(device)
    refusal_vector = refusal_vector.to(device)
    
    X = H_h_layer @ P_layer
    A = X.T @ X + lambda_reg * (P_layer.T @ P_layer)
    b = X.T @ refusal_vector.repeat(X.shape[0], 1)
    tilde_delta_layer = torch.linalg.pinv(A) @ b
    
    result = X @ tilde_delta_layer
    avg_reconstruction_error = torch.norm(
        result - refusal_vector.repeat(X.shape[0], 1)
    ) / X.shape[0]
    
    logger.info(f"avg_reconstruction_error: {avg_reconstruction_error:.6f}\t"
                f"refusal_vector norm: {torch.norm(refusal_vector):.6f}")
    
    return tilde_delta_layer


def cal_steering_matrix_l(P_layer, tilde_delta_layer, device="cuda:0"):
    P_layer = P_layer.to(device)
    tilde_delta_layer = tilde_delta_layer.to(device)
    
    steering_matrix_layer = P_layer @ tilde_delta_layer
    
    return steering_matrix_layer


def compute_steering_matrix_layer(
    H_benign: torch.Tensor,
    H_malicious: torch.Tensor,
    risk_vector: torch.Tensor,
    nullspace_ratio: float = 0.6,
    lambda_reg: float = 10.0,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H_benign = H_benign.to(device).float()
    H_malicious = H_malicious.to(device).float()
    risk_vector = risk_vector.to(device).float()
    
    logger.info(f"H_benign shape: {H_benign.shape}")
    logger.info(f"H_malicious shape: {H_malicious.shape}")
    logger.info(f"risk_vector shape: {risk_vector.shape}")
    
    logger.info("Computing null-space projection matrix...")
    P = null_space_projection_l(H_benign, abs_nullspace_ratio=nullspace_ratio)
    
    logger.info("Computing transformation matrix...")
    Delta_tilde = cal_tilde_delta_with_regularization_l(
        H_malicious, P, risk_vector, lambda_reg=lambda_reg, device=device
    )
    
    steering_matrix = cal_steering_matrix_l(P, Delta_tilde, device=device)
    
    logger.info(f"P norm: {torch.norm(P).item():.4f}")
    logger.info(f"Delta_tilde norm: {torch.norm(Delta_tilde).item():.4f}")
    logger.info(f"Steering matrix norm: {torch.norm(steering_matrix).item():.4f}")
    
    test_benign = H_benign[:min(10, H_benign.shape[0]), :]
    steering_on_benign = test_benign @ steering_matrix.T
    benign_steering_norm = torch.norm(steering_on_benign).item()
    logger.info(f"Benign sample steering vector norm (should be close to 0): {benign_steering_norm:.6f}")
    
    test_malicious = H_malicious[:min(10, H_malicious.shape[0]), :]
    steering_on_malicious = test_malicious @ steering_matrix.T
    risk_vector_expanded = risk_vector.unsqueeze(0).expand(test_malicious.shape[0], -1)
    alignment = torch.nn.functional.cosine_similarity(
        steering_on_malicious, 
        risk_vector_expanded, 
        dim=1
    ).mean().item()
    logger.info(f"Malicious sample steering cosine similarity with risk vector (should be close to 1): {alignment:.4f}")
    
    return P, Delta_tilde, steering_matrix
