set -x
export CUDA_VISIBLE_DEVICES=5

MODEL_PATH=/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct #/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct
export TASK=musique  # 2wikimultihop  hotpot
export MODEL_NAME=qwen_7b_debug
train_files="['data/${TASK}/train_false.parquet']"
test_files="['data/${TASK}/test_false.parquet']"

export NLTK_DATA=/mnt/shared-storage-user/liyafu/runquan/nltk_data
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled 
export RAY_memory_usage_threshold=0.98

export USE_LLM_JUDGE=true
export LLM_JUDGE_API_BASE=http://localhost:8000
export LLM_JUDGE_MODEL_NAME=/mnt/shared-storage-user/liyafu/models/Llama-3.3-70B-Instruct
export LLM_JUDGE_API_KEY=  # Empty or your API key
export LLM_JUDGE_MAX_WORKERS=8  # Number of concurrent judge requests
export LLM_JUDGE_TIMEOUT=30  # Timeout per request (seconds)

export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export RAY_DISABLE_ENV_HOOK=1

export FOUNDATION_LOCAL_PATH=/mnt/shared-storage-user/liyafu/runquan/models/flan-t5-base
export STRATEGY=grpo
export SENTENCE_LAMBDA=0.5
export SENTENCE_LAMBDA_POS=0.5
export SENTENCE_LAMBDA_NEG=0.5
export GRPO_VARIANCE_THRESHOLD=0

export ENABLE_SYSTEM_PROMPT_INJECTION=true
export MODEL_TEMPLATE=qwen
export SYSTEM_PROMPT_TYPE=rlcr #idk_aware, idk_not_aware, rlcr
export USE_REWARD=RLCR  #GRPO, TruthRL, THS, RLCR

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    +data.max_samples=6000 \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    +actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.n=1 \
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
    trainer.experiment_name=$STRATEGY \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir="checkpoints/\${trainer.project_name}/${MODEL_NAME}/\${trainer.experiment_name}" \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@ 2>&1 | tee ${STRATEGY}_${TASK}_debug.log
    #trainer.resume_mode='checkpoints/musique/qwen_7b/grpo/global_step_230' \