#!/bin/bash
set -e

DATASET=MuSiQue
# dataset for 2WikimultihopQA, HotpotQA, MuSiQue, Halueval
TYPE=test
# type for train, test
#python preprocess.py --dataset hotpot --type test --size 500 --seed 123
# Run for specific dataset and type
echo "Processing $DATASET ($TYPE)..."
python preprocess.py --dataset "$DATASET" --type "$TYPE" --size 2000
