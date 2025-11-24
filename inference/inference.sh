set -x

export CUDA_VISIBLE_DEVICES=0
MODEL_NAME=qwen
export MODEL_TEMPLATE=qwen
DATASET=halueval
MODEL_PATH=/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct
#/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct
#Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct
test_files="['data/$DATASET/test.parquet']"
export SYSTEM_PROMPT_TYPE=tot #directly, cot, tot, dac, htp

OUTPUT_FILE=inference/inference_results/$MODEL_NAME/$DATASET.jsonl
FILTER_TYPE=all  # all, answerable, unanswerable
NUM_SAMPLES=-1  # -1 表示处理全部
MAX_MODEL_LEN=16384  # 模型最大上下文长度（输入+输出总容量）
TENSOR_PARALLEL_SIZE=1

TEMPERATURE=0
TOP_P=1.0  # Ignored in greedy mode
TOP_K=-1   # Ignored in greedy mode  
REPETITION_PENALTY=1.0  # No repetition penalty in validation
MAX_TOKENS=4096  # 单次生成的最大输出长度

# LLM Judge configuration (must match server configuration)
export USE_LLM_JUDGE=true
export LLM_JUDGE_API_BASE=http://100.103.112.35:8000
export LLM_JUDGE_MODEL_NAME=/mnt/shared-storage-user/liyafu/models/Llama-3.3-70B-Instruct
export LLM_JUDGE_API_KEY=
export LLM_JUDGE_MAX_WORKERS=8
export LLM_JUDGE_TIMEOUT=60

export IDK=true
python inference/inference.py \
    --test-files "$test_files" \
    --output-dir "$OUTPUT_FILE" \
    --model-path $MODEL_PATH \
    --model-name $MODEL_NAME \
    --filter-type $FILTER_TYPE \
    --num-samples $NUM_SAMPLES \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --repetition-penalty $REPETITION_PENALTY \
    --max-tokens $MAX_TOKENS \
    ${CHECKPOINT_PATH:+--checkpoint-path "$CHECKPOINT_PATH"} \
    $@