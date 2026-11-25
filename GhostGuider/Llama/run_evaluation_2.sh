#!/bin/bash

MODEL_PATH="/path-to-model/Llama-3.2-11B-Vision-Instruct"
STEERING_CHECKPOINT="/path-to-code/MRIS/GhostGuider/Llama/outputs/llama_dual_protected_steering/steering_matrices_instruct.pt"

INPUT_FILE="/path-to-code/MRIS/data/MRISbench/MRISbench_test.json"

OUTPUT_DIR="./results"
RECORD_DIR="${OUTPUT_DIR}/records"
LOGS_DIR="${OUTPUT_DIR}/logs"
STATS_DIR="${OUTPUT_DIR}/stats"

mkdir -p "${RECORD_DIR}" "${LOGS_DIR}" "${STATS_DIR}"

echo "Running evaluations in parallel..."

CUDA_VISIBLE_DEVICES=0 python eval_llama_with_steer_2.py \
    --model_path "${MODEL_PATH}" \
    --steering_checkpoint "${STEERING_CHECKPOINT}" \
    --input_file "${INPUT_FILE}" \
    --model_name "LlamaVision_steering_0.6" \
    --output_record_dir "${RECORD_DIR}" \
    --output_logs_dir "${LOGS_DIR}" \
    --output_record_stats "${STATS_DIR}" \
    --steering_layers "19-39" \
    --steering_strength 0.6 \
    > "${LOGS_DIR}/LlamaVision_steering_0.6.log" 2>&1 &

PID_EXAMPLE2=$!
echo "Example 2 started with PID: ${PID_EXAMPLE2}"
echo "All evaluations complete!"
echo "Results saved to: ${OUTPUT_DIR}"
