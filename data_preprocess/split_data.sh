TYPE=train #test
DATASET=hotpot

python data_preprocess/split_data.py \
    --input /mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/data/$DATASET/$TYPE.parquet \
    --output_true /mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/data/$DATASET/${TYPE}_true.parquet \
    --output_false /mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/data/$DATASET/${TYPE}_false.parquet