set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_CONFIG="\
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.return_raw_input_ids=True \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=1"

# Define the ppl values for curriculum learning
PPL_VALUES="1"

# Initial model path
MODEL_PATH=/home/project/11004114/rl/llm/Qwen2.5-1.5B-Instruct
REWARD_MODEL_PATH=/home/project/11004114/rl/llm/HHEM
PROJECT_NAME=GRPO_ASQA
EXPERIMENT_NAME=GRPO-Qwen-7B-Instruct-curriculum

for ppl in $PPL_VALUES; do
    echo "Starting training for ${ppl}ans"

    TRAIN_FILE="data/asqa/qwen-instruct/${ppl}ans/train.parquet"
    VAL_FILE="data/asqa/qwen-instruct/test.parquet" # standard bench

    # Define experiment name for this stage
    LOG_FILE="grpo_qwen_7b_instruct_curriculum_${ppl}ans.log"

    # Construct the command
    COMMAND="python3 -m verl.trainer.main_ppo \
        ${BASE_CONFIG} \
        data.train_files=${TRAIN_FILE} \
        data.val_files=${VAL_FILE} \
        reward_model.model.path=${REWARD_MODEL_PATH} \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        $@ 2>&1 | tee ${LOG_FILE}"

    echo "Executing command: ${COMMAND}"
    eval ${COMMAND}
done

echo "Curriculum learning finished!"
