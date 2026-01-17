#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=4,5
# 添加当前目录到 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SYSTEM_PROMPT_TYPE=cot

# 设置训练资源
NNODES=1
N_GPUS=2  # 使用2张GPU

# 模型和数据路径
MODEL_PATH="/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct"
# TRAIN_FILE 和 VAL_FILE 支持:
# 1. 单个文件路径: "path/to/file.parquet"
# 2. 通配符模式: "path/to/*.parquet"
# 3. 文件列表字符串: "['path/to/file1.parquet', 'path/to/file2.parquet']"
TRAIN_FILE="SFT/sft_data/qwen/cosmo/HotpotQA/*.parquet"
VAL_FILE=null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 输出目录和日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="SFT/output/sft_experiment_$TIMESTAMP"
LOG_FILE="$OUTPUT_DIR/train.log"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

echo "开始训练，日志将输出到: $LOG_FILE"

# 使用 nohup 在后台启动训练，输出重定向到日志文件
nohup torchrun --nproc_per_node=$N_GPUS \
    verl/trainer/fsdp_sft_trainer.py \
    --config-path "config" \
    --config-name "sft_trainer" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.max_length=8192 \
    model.partial_pretrain="$MODEL_PATH" \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir="$OUTPUT_DIR" \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    trainer.total_epochs=3 \
    optim.lr=2e-5 \
    > "$LOG_FILE" 2>&1 &

# 输出进程ID
echo "训练已在后台启动，进程ID: $!"
echo "可以使用以下命令查看日志: tail -f $LOG_FILE"
