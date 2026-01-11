#!/bin/bash
set -e

# Model and Data Config
export CUDA_VISIBLE_DEVICES=5
MODEL_PATH="/mnt/shared-storage-user/liyafu/models/Llama-3.1-8B-Instruct" #Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct 
DATASET=MuSiQue  #2WikimultihopQA, HotpotQA, Halueval, MuSiQue
DATASET_PATH="data/${DATASET}/test.parquet" 
MODEL_TEMPLATE=llama #qwen, llama
OUTPUT_DIR="output/inference/${DATASET}/${MODEL_TEMPLATE}"
PROMPT_TEMPLATE="cod"  # cot, directly, tot, htp, cod, tale


# LLM Judge Config
export LLM_JUDGE_API_BASE="http://localhost:8000"
export LLM_JUDGE_MODEL_NAME="/mnt/shared-storage-user/liyafu/models/Llama-3.3-70B-Instruct"
export LLM_JUDGE_API_KEY=""
export LLM_JUDGE_MAX_WORKERS=8
export LLM_JUDGE_TIMEOUT=60
TEMPERATURE=0
MAX_OUTPUT=2048
# Run Inference
python3 inference.py \
    --model "$MODEL_PATH" \
    --datasets "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --max-output "$MAX_OUTPUT" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --model-template "$MODEL_TEMPLATE" \
    --judge-api-base "$LLM_JUDGE_API_BASE" \
    --judge-model-name "$LLM_JUDGE_MODEL_NAME" \
    --judge-api-key "$LLM_JUDGE_API_KEY" \
    --judge-max-workers "$LLM_JUDGE_MAX_WORKERS" \
    --judge-timeout "$LLM_JUDGE_TIMEOUT"

