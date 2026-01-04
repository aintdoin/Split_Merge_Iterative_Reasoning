set -x

export CUDA_VISIBLE_DEVICES=4

# Input/Output Configuration
DATASET=musique
INPUT_FILE="data/${DATASET}/train.parquet"
OUTPUT_FILE="SFT/data/rtuning_sft_train.parquet"
MODEL_PATH="/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct"
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=16384
MODEL_TEMPLATE=qwen

mkdir -p $(dirname "$OUTPUT_FILE")

python SFT/rtuning_for_sft.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --model-template $MODEL_TEMPLATE

