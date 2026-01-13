#!/bin/bash
set -x

# Environment Configuration
export CUDA_VISIBLE_DEVICES=4,5

# Method Selection: cosmo or C3oT
export USE_METHOD=cosmo

# LLM Judge Settings
export USE_LLM_JUDGE=true
export LLM_JUDGE_API_BASE=http://localhost:8000
export LLM_JUDGE_MODEL_NAME=/mnt/shared-storage-user/liyafu/models/Llama-3.3-70B-Instruct
export LLM_JUDGE_API_KEY=  # Empty or your API key
export LLM_JUDGE_MAX_WORKERS=8  # Number of concurrent judge requests
export LLM_JUDGE_TIMEOUT=300  # Timeout per request (seconds)

# File Paths
export INPUT_FILE="output/inference/HotpotQA/qwen/results.jsonl"
export OUTPUT_FILE="SFT/sft_data/qwen/${USE_METHOD}/HotpotQA.parquet"
export MAX_SAMPLES=20000

# Execution
echo "Running Data Processing with Method: $USE_METHOD"

python SFT/sft_data.py \
    --method $USE_METHOD \
    --input $INPUT_FILE \
    --output $OUTPUT_FILE \
    --max_samples $MAX_SAMPLES
