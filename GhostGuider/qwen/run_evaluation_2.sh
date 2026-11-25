#!/bin/bash

MODEL_PATH="/path-to-model/Qwen2.5-VL-7B-Instruct"
STEERING_CHECKPOINT="./outputs/qwen_dual_protected_steering/steering_matrices.pt"

INPUT_FILE="/path-to-code/MRIS/data/MRISbench/MRISbench_test.json"

OUTPUT_DIR="./results"
RECORD_DIR="${OUTPUT_DIR}/records"
LOGS_DIR="${OUTPUT_DIR}/logs"
STATS_DIR="${OUTPUT_DIR}/stats"

mkdir -p "${RECORD_DIR}" "${LOGS_DIR}" "${STATS_DIR}"

echo "Running evaluations in parallel..."

CUDA_VISIBLE_DEVICES=2 python eval_qwen_with_steer_2.py \
    --model_path "${MODEL_PATH}" \
    --steering_checkpoint "${STEERING_CHECKPOINT}" \
    --input_file "${INPUT_FILE}" \
    --model_name "qwen2.5_steering_0.6_all" \
    --output_record_dir "${RECORD_DIR}" \
    --output_logs_dir "${LOGS_DIR}" \
    --output_record_stats "${STATS_DIR}" \
    --steering_layers "14-27" \
    --steering_strength 0.6 \
    > "${LOGS_DIR}/qwen2.5_steering_0.6_all.log" 2>&1 &

PID=$!
echo "Evaluation started with PID: ${PID}"

echo "Waiting for evaluation to complete..."
wait ${PID}
echo "Evaluation completed"

echo "All evaluations complete!"
echo "Results saved to: ${OUTPUT_DIR}"
