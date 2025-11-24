# 默认参数
export CUDA_VISIBLE_DEVICES=4
export USE_LLM_JUDGE=true  # Enable LLM judge
export LLM_JUDGE_API_BASE=http://localhost:8000
export LLM_JUDGE_MODEL_NAME=/mnt/shared-storage-user/liyafu/models/Llama-3.3-70B-Instruct
export LLM_JUDGE_API_KEY=  # Empty or your API key
MODEL_PATH="/mnt/shared-storage-user/liyafu/models/Qwen2.5-7B-Instruct"
N_CANDIDATES=10
UPPER=0.8
LOWER=0.2
SAMPLE_SIZE=10000
DATA_PATH="/mnt/shared-storage-user/liyafu/runquan/musique/data/musique_full_v1.0_train.jsonl"
OUTPUT_PATH="data/musique_filter.parquet"
TEMPLATE_TYPE="qwen"

# 解析命令行参数
while getopts "m:n:u:l:d:o:t:" opt; do
  case $opt in
    m) MODEL_PATH="$OPTARG" ;;
    n) N_CANDIDATES="$OPTARG" ;;
    u) UPPER="$OPTARG" ;;
    l) LOWER="$OPTARG" ;;
    d) DATA_PATH="$OPTARG" ;;
    o) OUTPUT_PATH="$OPTARG" ;;
    t) TEMPLATE_TYPE="$OPTARG" ;;
    *) echo "用法: $0 [-m model_path] [-n n_candidates] [-u upper] [-l lower] [-d data_path] [-o output_path] [-t template_type]" ; exit 1 ;;
  esac
done

# 创建日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 日志文件名，包含时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/filter_${TIMESTAMP}.log"

# 运行Python脚本（后台运行）
echo "开始执行过滤..."
echo "输出将记录到: $LOG_FILE"
nohup python "$(dirname "$0")/data_filter.py" \
  --data-path "$DATA_PATH" \
  --output-path "$OUTPUT_PATH" \
  --model-path "$MODEL_PATH" \
  --n-candidates "$N_CANDIDATES" \
  --upper "$UPPER" \
  --lower "$LOWER" \
  --sample-size "$SAMPLE_SIZE" \
  --temperature 1.0 \
  --do-sample True \
  --top-p 1.0 \
  --top-k 50 \
  --max-tokens 2048 \
  --template-type "$TEMPLATE_TYPE" > "$LOG_FILE" 2>&1 &

# 保存进程ID
PID=$!
echo "进程ID: $PID"
echo "$PID" > "$LOG_DIR/current_filter.pid"

echo "过滤任务已在后台启动，请使用 'tail -f $LOG_FILE' 查看实时输出"
echo "或使用 'kill $PID' 终止任务"