#!/bin/bash
set -x

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0

# Method Selection
# Options: cosmo, C3oT, FS_BoN, SPIRIT
export USE_METHOD=cosmo

# Dataset Configuration
DATASET=2wikimultihop
export INPUT_FILE=data/${DATASET}/train.parquet
export OUTPUT_FILE=SFT/data/${DATASET}_train.parquet
export OUTPUT_RL_FILE=data/${DATASET}/train_rl.parquet

# Model Configuration (Required for C3oT, FS_BoN, SPIRIT)
# Point to the model used for generating reasoning traces and scoring
export MODEL_PATH=/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct

# Inference Parameters
export TENSOR_PARALLEL_SIZE=1
export MAX_MODEL_LEN=16384
export SYSTEM_PROMPT_TYPE=cot
export MAX_TOKENS=4096
export GPU_MEMORY_UTILIZATION=0.9

# LLM Judge Configuration (Required for C3oT and FS_BoN)
# Reference: inference/inference.sh
export USE_LLM_JUDGE=true
export LLM_JUDGE_API_BASE=http://localhost:8000/v1
export LLM_JUDGE_MODEL_NAME=/mnt/shared-storage-user/liyafu/models/Llama-3.3-70B-Instruct
export LLM_JUDGE_API_KEY=EMPTY
export LLM_JUDGE_MAX_WORKERS=8
export LLM_JUDGE_TIMEOUT=60

# Run the processing script
python SFT/process_data_for_reasoning.py
