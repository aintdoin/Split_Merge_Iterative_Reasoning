export CUDA_VISIBLE_DEVICES=5
LOCAL_DIR="/mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/checkpoints/musique/qwen_7b/grpo_evar_math_scaled/global_step_380/actor"

python scripts/model_merger.py --local_dir "$LOCAL_DIR"