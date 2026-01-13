#!/bin/bash
set -e

# Model and Data Config
export CUDA_VISIBLE_DEVICES=4
MODEL_PATH=SFT/output/sft_experiment_20260113_004846/global_step_936  #"/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct" #Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct
DATASET=MuSiQue  #2WikimultihopQA, HotpotQA, Halueval, MuSiQue
DATASET_PATH="data/${DATASET}/test.parquet"  #data/MuSiQue/hops_split/4hop_1000.parquet
MODEL_TEMPLATE=qwen #qwen, llama
OUTPUT_DIR="output/inference/${DATASET}/${MODEL_TEMPLATE}/C3oT"    #debug
PROMPT_TEMPLATE="cot"  # cot, directly, tot, htp, cod, tale


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

