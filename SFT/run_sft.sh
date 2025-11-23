#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=0,1
# 添加当前目录到 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SYSTEM_PROMPT_TYPE=directly

# 设置训练资源
NNODES=1
N_GPUS=2  # 使用2张GPU

# 模型和数据路径
MODEL_PATH="/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct"
TRAIN_FILE="SFT/data/musique_for_sft.parquet"
VAL_FILE="data/musique/test_true.parquet"

# 输出目录
OUTPUT_DIR="SFT/output/sft_experiment_$(date +%Y%m%d_%H%M%S)"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 启动训练
# 使用 hydra 的语法加载配置文件并覆盖特定参数
torchrun --nproc_per_node=$N_GPUS \
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
    data.micro_batch_size_per_gpu=1 \
    trainer.total_epochs=3 \
    optim.lr=2e-5
