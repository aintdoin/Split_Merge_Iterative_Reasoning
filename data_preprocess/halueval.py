""" Preprocess dataset for HaluEval QA task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import random


def make_prefix(dp, template_type):
    question = dp.get('question', 'no question')
    knowledge = dp.get('knowledge', 'No knowledge provided.')
    right_answer = dp.get('right_answer', '')
    hallucinated_answer = dp.get('hallucinated_answer', '')
    
    # Randomly shuffle the two candidate answers
    candidate_answers = [right_answer, hallucinated_answer]
    random.shuffle(candidate_answers)
    
    # Format knowledge as references
    references_context = f"Knowledge: {knowledge}"
    
    # Define the user content block - use unified format to avoid model outputting references
    user_content = f"""**References:**
{references_context}

**Question:**
{question}

**Candidate Answers:**
- {candidate_answers[0]}
- {candidate_answers[1]}"""
    return user_content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--size', type=int, default=5000)
    parser.add_argument('--template_type', type=str, default='qwen')
    parser.add_argument('--data_path', default='/mnt/shared-storage-user/liyafu/runquan/HaluEval/data/qa_data.json')
    parser.add_argument('--local_dir', default=None)
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'HaluEval'
    
    # Set local_dir based on template_type if not provided
    if args.local_dir is None:
        args.local_dir = f'data/halueval/{args.template_type}'

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)

    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(f"Total dataset length: {len(raw_dataset)}")

    # Split dataset based on type
    if args.type == 'train':
        TRAIN_SIZE = args.size
        assert len(raw_dataset) >= TRAIN_SIZE
        dataset = raw_dataset.select(range(TRAIN_SIZE))
        print(f"Train dataset length: {len(dataset)}")
    else:
        TEST_SIZE = args.size
        # For test, we start from a different offset to avoid overlap with train
        start_idx = 5000  # Assuming train uses first 5000
        assert len(raw_dataset) >= start_idx + TEST_SIZE
        dataset = raw_dataset.select(range(start_idx, start_idx + TEST_SIZE))
        print(f"Test dataset length: {len(dataset)}")

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt_text = make_prefix(example, template_type=args.template_type)
            question = example.get('question', '')
            answer = example.get('right_answer', '')
            
            data = {
                "prompt": prompt_text,
                "question": question,
                "answer": answer,
                "data_source": data_source,
                "extra_info": {
                    'index': idx,
                    'sample_id': f'haluevalqa_{idx}',
                    'answerable': True,
                }
            }
            return data

        return process_fn

    # Remove original columns after they've been formatted into prompt
    cols_to_remove = [col for col in dataset.column_names if col not in ['prompt', 'question', 'answer', 'data_source', 'extra_info']]
    dataset = dataset.map(function=make_map_fn(args.type), with_indices=True, remove_columns=cols_to_remove)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    # Save based on type
    if args.type == 'train':
        dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
        print(f"Saved train dataset to {os.path.join(local_dir, 'train.parquet')}")
    else:
        dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
        print(f"Saved test dataset to {os.path.join(local_dir, 'test.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
