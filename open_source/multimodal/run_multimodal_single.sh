#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python -u ./utils/qwen_process_mris_hybrid_multi.py \
  --model_path /path-to-model/Qwen/Qwen2.5-VL-7B-Instruct \
  --input_file /path-to-code/MRIS/data/MRISbench/MRISbench.json \
  --model_name Qwen2.5-VL-7B-Instruct \
  --output_record_dir /path-to-code/MRIS/data/multimodal/records \
  --output_logs_dir /path-to-code/MRIS/data/multimodal/logs \
  --output_record_stats /path-to-code/MRIS/data/multimodal/stats \
  --temperature 0.0 \
  --max_new_tokens 100 \
  --top_p 0.9 \
  --save_interval 100 

CUDA_VISIBLE_DEVICES=2 python -u ./utils/llama_vision_process_mris_hybrid_multi.py \
  --model_path /path-to-model/Llama-Vision/Llama-3.2-11B-Vision-Instruct \
  --input_file /path-to-code/MRIS/data/MRISbench/MRISbench.json \
  --model_name Llama-3.2-11B-Vision-Instruct \
  --output_record_dir /path-to-code/MRIS/data/multimodal/records \
  --output_logs_dir /path-to-code/MRIS/data/multimodal/logs \
  --output_record_stats /path-to-code/MRIS/data/multimodal/stats \
  --temperature 0.0 \
  --max_new_tokens 100 \
  --top_p 0.9 \
  --save_interval 100  