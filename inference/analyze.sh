#!/bin/bash
set -x
MODEL_NAME=qwen-true # deepseek-qwen, qwen, qwen-musique-tuned
DATASET=hotpot # 2wikimultihop, halueval, hotpot, musique

INPUT_FILE="inference/inference_results/$MODEL_NAME/$DATASET.jsonl"
#THINK_OFF_FILE="inference/inference_results/$MODEL_NAME/$DATASET-think_off.jsonl"

# 输出目录
OUTPUT_DIR="inference/results_length/$MODEL_NAME"

# 输出文件名前缀（默认留空，会自动从输入文件名提取）
PREFIX=""

python inference/analyze.py \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    ${PREFIX:+--prefix "$PREFIX"} \
    ${THINK_OFF_FILE:+--think-off-file "$THINK_OFF_FILE"}

