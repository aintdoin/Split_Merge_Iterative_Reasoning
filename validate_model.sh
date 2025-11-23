#!/bin/bash
set -x

# Model paths
MODEL_PATH=/mnt/shared-storage-user/liyafu/runquan/models/DeepSeek-R1-Distill-Qwen-7B
REWARD_MODEL_PATH=/mnt/shared-storage-user/liyafu/runquan/models/HHEM

# Checkpoint to validate (modify this to your checkpoint path)
#CHECKPOINT_PATH='checkpoints/mixed/GRPO/global_step_1100'

# Task configuration
export TASK=mixed  # 2wikimultihop  musique_ans  hotpot mixed 2wikimultihop-shuffle

# Data files
train_files="['data/2wikimultihop/deepseek-r1-distill-qwen/train.parquet', 'data/musique_ans/deepseek-r1-distill-qwen/train.parquet', 'data/hotpot/deepseek-r1-distill-qwen/train.parquet']"
test_files="['data/2wikimultihop-shuffle/deepseek-r1-distill-qwen/test.parquet', 'data/musique_ans-shuffle/deepseek-r1-distill-qwen/test.parquet', 'data/hotpot-shuffle/deepseek-r1-distill-qwen/test.parquet']"

# Environment variables
export NLTK_DATA=/mnt/shared-storage-user/liyafu/runquan/nltk_data
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled 
export RAY_memory_usage_threshold=0.98

# LLM Judge Configuration
export USE_LLM_JUDGE=false  # Set to false to use F1 score instead of LLM judge
# If using LLM judge, configure these:
# export USE_LLM_JUDGE=true
# export LLM_JUDGE_API_BASE=http://localhost:8000
# export LLM_JUDGE_MODEL_NAME=/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct
# export LLM_JUDGE_API_KEY=
# export LLM_JUDGE_MAX_WORKERS=8
# export LLM_JUDGE_TIMEOUT=30

# Legacy settings (disabled)
export USE_LLM_ANSWER_EXTRACTION=false
export ENABLE_BATCH_ANSWER_PROCESSING=false

export FOUNDATION_LOCAL_PATH=/mnt/shared-storage-user/liyafu/runquan/models/flan-t5-base
export STRATEGY=grpo_evar_sentence_scaled
export SENTENCE_LAMBDA=0.2

# Build resume mode argument if checkpoint provided
if [ -n "$CHECKPOINT_PATH" ]; then
    RESUME_ARG="trainer.resume_mode='$CHECKPOINT_PATH'"
    echo "Loading checkpoint from: $CHECKPOINT_PATH"
else
    RESUME_ARG=""
    echo "No checkpoint specified, using base model: $MODEL_PATH"
fi

# Run validation only (no training)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=22200 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=50 \
    +actor_rollout_ref.rollout.repetition_penalty=1.1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=True \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.micro_batch_size_per_gpu=64 \
    reward_model.model.trust_remote_code=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$TASK \
    trainer.experiment_name='VALIDATION_ONLY' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir='validation_results/${trainer.project_name}' \
    trainer.default_hdfs_dir=null \
    +trainer.val_only=True \
    $RESUME_ARG \
    $@

