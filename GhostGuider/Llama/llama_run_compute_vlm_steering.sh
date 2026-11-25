#!/bin/bash

set -e

echo "=========================================="
echo "Dual-Protected GhostGuider for Llama-Vision"
echo "=========================================="

MODEL_PATH="/path-to-model/Llama-Vision/Llama-3.2-11B-Vision-Instruct"
OUTPUT_DIR="outputs/llama_dual_protected_steering"
DEVICE="cuda"

BENIGN_DATA="/path-to-code/MRIS/data/benign.json"
HARMFUL_DATA="/path-to-code/MRIS/data/malicious.json"
MRIS_DATA="/path-to-code/MRIS/data/MRISbench/MRISbench.json"
RISK_VECTORS="/path-to-code/GhostGuider/Llama/risk_vector/risk_vectors_llama32_instruct_mean_diff.npy"

BENIGN_SAMPLE_RATIO=0.4
HARMFUL_SAMPLE_RATIO=0.4
MALICIOUS_SAMPLE_RATIO=0.4

LAYERS_CONFIG="19-39:0.6"
NULLSPACE_RATIO=0.6
LAMBDA_REG=10.0
BATCH_SIZE=4

mkdir -p $OUTPUT_DIR

echo ""
echo "=========================================="
echo "Computing Dual-Protected Steering Matrices"
echo "=========================================="
echo "Configuration:"
echo "  - Model path: $MODEL_PATH"
echo "  - Benign sample ratio: ${BENIGN_SAMPLE_RATIO} (${BENIGN_SAMPLE_RATIO%.*}%)"
echo "  - Harmful sample ratio: ${HARMFUL_SAMPLE_RATIO} (${HARMFUL_SAMPLE_RATIO%.*}%)"
echo "  - MRIS sample ratio: ${MALICIOUS_SAMPLE_RATIO} (${MALICIOUS_SAMPLE_RATIO%.*}%)"
echo "  - Layer config: $LAYERS_CONFIG"
echo "  - Lambda regularization: $LAMBDA_REG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python llama_compute_vlm_steering_matrix.py \
    --model_path "$MODEL_PATH" \
    --benign_data "$BENIGN_DATA" \
    --harmful_data "$HARMFUL_DATA" \
    --mris_data "$MRIS_DATA" \
    --risk_vectors "$RISK_VECTORS" \
    --output_path "$OUTPUT_DIR/steering_matrices_instruct.pt" \
    --layers_config "$LAYERS_CONFIG" \
    --nullspace_ratio $NULLSPACE_RATIO \
    --lambda_reg $LAMBDA_REG \
    --benign_sample_ratio $BENIGN_SAMPLE_RATIO \
    --harmful_sample_ratio $HARMFUL_SAMPLE_RATIO \
    --malicious_sample_ratio $MALICIOUS_SAMPLE_RATIO \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE"

echo ""
echo "=========================================="
echo "✅ Completed!"
echo "=========================================="
echo "Output files:"
echo "  - Steering matrices: $OUTPUT_DIR/steering_matrices.pt"
echo "  - Full information: $OUTPUT_DIR/steering_matrices_full.pt"
echo "  - Metadata: $OUTPUT_DIR/steering_matrices_metadata.json"
echo ""
echo "Next steps:"
echo "  1. Use steering_matrices.pt for steering during inference"
echo "  2. Add steering to VLM model's forward hook"
echo "  3. Evaluate model performance on benign/harmful/MRIS data"
echo "=========================================="
