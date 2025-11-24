set -euo pipefail

# 1. 新增：指定日志文件路径（可自定义，建议放在固定目录）
LOG_FILE="${LOG_FILE:-./logs/LLM_as_a_Judge_server.log}"

# 2. 原有变量配置（不变）
MODEL_PATH=/mnt/shared-storage-user/liyafu/models/Llama-3.3-70B-Instruct
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096} 
GPU_UTIL=${GPU_UTIL:-0.85}
export CUDA_VISIBLE_DEVICES=4,5,6,7
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 新增：批处理参数用于提升GPU利用率
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}  # 最大批处理序列数
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}  # 最大批处理token数

# 3. 原有参数检查（不变）
if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is required" >&2
  exit 1
fi

# 4. 新增：提示日志位置
echo "Starting vLLM OpenAI server on ${HOST}:${PORT} with model ${MODEL_PATH}..."
echo "Using ${TENSOR_PARALLEL_SIZE} GPUs for tensor parallelism (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Logs will be saved to: ${LOG_FILE}"
echo "Batch settings: max_num_seqs=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"

# 5. 修改：用 nohup 启动进程，并定向日志，添加批处理参数和张量并行
exec nohup python3 -m vllm.entrypoints.openai.api_server \
  --host ${HOST} \
  --port ${PORT} \
  --model ${MODEL_PATH} \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization ${GPU_UTIL} \
  --dtype bfloat16 \
  --trust-remote-code \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} > "${LOG_FILE}" 2>&1 &


