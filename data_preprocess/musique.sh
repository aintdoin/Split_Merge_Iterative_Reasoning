# type: train, test
# template_type: deepseek-r1-distill-qwen, deepseek-r1-distill-llama, base, qwen-instruct, llama-instruct
export LLM_JUDGE_API_BASE=http://localhost:8000
export LLM_JUDGE_MODEL_NAME=/mnt/shared-storage-user/liyafu/models/Llama-3.3-70B-Instruct
export LLM_JUDGE_API_KEY=
python -m data_preprocess.musique --type test --size 2000

python -m data_preprocess.musique   --type train --size 5000 --log-every 100