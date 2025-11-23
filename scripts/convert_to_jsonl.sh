#!/bin/bash

# 转换JSON到JSONL格式
#python /mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/examples/data_preprocess/convert_to_jsonl.py \
#    --input /mnt/shared-storage-user/liyafu/runquan/hotpot/hotpot_train_v1.1.json \
#    --output /mnt/shared-storage-user/liyafu/runquan/hotpot/hotpot_train_v1.1.jsonl

python /mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/scripts/convert_to_jsonl.py \
    --input /mnt/shared-storage-user/liyafu/runquan/musique_ans/data/dev.json \
    --output /mnt/shared-storage-user/liyafu/runquan/musique_ans/data/dev.jsonl