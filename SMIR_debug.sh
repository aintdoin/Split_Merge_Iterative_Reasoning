set -x
# /mnt/shared-storage-user/liyafu/models/DeepSeek-R1-Distill-Qwen-7B /mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct
MODEL_PATH=SFT/output/sft
# MiniCheck model is now served via API, path not directly used by training
#REWARD_MODEL_PATH=/mnt/shared-storage-user/liyafu/runquan/models/MiniCheck

export MODEL_NAME=qwen_7b  # Model identifier for checkpoint organization
export STRATEGY=cosmo #grpo, cosmo, lcpo, think_prune
export MODEL_TEMPLATE=qwen
export BETA_WARMUP_STEPS=100
export SYSTEM_PROMPT_TYPE=cot #cot directly
train_files="['data/musique/train_rl.parquet', 'data/2wikimultihop/train_rl.parquet']"
test_files="['data/musique/test.parquet', 'data/2wikimultihop/test.parquet', 'data/hotpot/test.parquet']"

export NLTK_DATA=/mnt/shared-storage-user/liyafu/runquan/nltk_data
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export RAY_memory_usage_threshold=0.98
export USE_LLM_JUDGE=true
export LLM_JUDGE_API_BASE=http://100.103.44.1:8000
export LLM_JUDGE_MODEL_NAME=/mnt/shared-storage-user/liyafu/models/Llama-3.3-70B-Instruct
export LLM_JUDGE_API_KEY=  # Empty or your API key
export LLM_JUDGE_MAX_WORKERS=8  # Number of concurrent judge requests
export LLM_JUDGE_TIMEOUT=60  # Timeout per request (seconds)


python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.grad_clip=0.5 \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.val_batch_size=512 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    +actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=True \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.micro_batch_size_per_gpu=32 \
    reward_model.model.trust_remote_code=True \
    data.shuffle=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=SMIR \
    +trainer.val_before_train=False \
    trainer.experiment_name=$STRATEGY \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir="checkpoints/\${trainer.project_name}/${MODEL_NAME}/\${trainer.experiment_name}" \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@ 2>&1 | tee ${STRATEGY}_${TASK}_debug.log
    #trainer.resume_mode='checkpoints/musique/qwen_7b/global_step_380' \





# MiniCheck API Configuration (replaces HHEM)
#export MINICHECK_API_BASE=http://localhost:8001
#export MINICHECK_MAX_WORKERS=8  # Number of concurrent API requests
#export MINICHECK_TIMEOUT=30  # Timeout per request (seconds)
# Dynamic IDK Penalty (自适应IDK惩罚，防止收敛到全部输出IDK)
#export ENABLE_DYNAMIC_IDK_PENALTY=false  # 启用动态IDK惩罚
#export IDK_PENALTY_ANSWERABLE=0  # 仅在动态惩罚禁用时使用的静态惩罚值
# Filter out unanswerable samples during training when set to true
#export disable_unanswerable=false
# Answer Length Penalty (限制answer长度的奖励惩罚)
#export ENABLE_ANSWER_LENGTH_PENALTY=false  # 启用长度惩罚
#export ANSWER_MAX_FREE_TOKENS=10           # 允许的最大token数（超过后开始惩罚）
#export ANSWER_PENALTY_PER_TOKEN=0.1        # 每超出1个token的惩罚值
#export ANSWER_MIN_FINAL_REWARD=-1.5        # 最终reward的下限（无论base reward是多少）
# Legacy (DISABLED - no longer used)
#export USE_LLM_ANSWER_EXTRACTION=false  # Disabled - we don't do LLM extraction anymore
#export ENABLE_BATCH_ANSWER_PROCESSING=false  # Disabled - not needed for judge-only mode
#export IDK_PENALTY_ANSWERABLE=-0.3 $@ 2>&1 | tee ${STRATEGY}_${TASK}_debug.log
    #trainer.resume_mode='checkpoints/musique/qwen_7b/grpo/global_step_230' \