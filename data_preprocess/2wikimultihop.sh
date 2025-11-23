# type: train, test
# template_type: deepseek-r1-distill-qwen, deepseek-r1-distill-llama, base, qwen-instruct, llama-instruct
export USE_LLM_JUDGE=true  # Enable LLM judge

# LLM Judge API Configuration (renamed from ANSWER_EXTRACT_* for clarity)
export LLM_JUDGE_API_BASE=http://100.103.112.35:8000
export LLM_JUDGE_MODEL_NAME=/mnt/shared-storage-user/liyafu/models/Llama-3.3-70B-Instruct
export LLM_JUDGE_API_KEY=  # Empty or your API key

python data_preprocess/2wikimultihop.py \
    --type train \
    --template_type qwen \
    --size 5000 \
    --data-path /mnt/shared-storage-user/liyafu/runquan/2wikimultihop/data/train.jsonl \
    --model-path /mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 4 \
    --n-candidates 32 \
    --temperature 1.0 --top-p 0.95 --top-k 100 --max-tokens 2048