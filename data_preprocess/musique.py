""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import random
import threading
import concurrent.futures

def make_prefix(dp):
    question = dp.get('question', 'no question')
    documents_str = dp.get('documents', '[]')
    
    # Parse and format the documents.
    import ast
    try:
        documents_list = ast.literal_eval(documents_str)
    except:
        documents_list = []
    
    formatted_docs = []
    for doc in documents_list:
        if isinstance(doc, list) and len(doc) == 2:
            title, content = doc
            # content could be a list of paragraphs or a single string
            if isinstance(content, list):
                text = ' '.join(str(item) for item in content)
            else:
                text = str(content)
            formatted_docs.append(f"Document '{title}': {text}")
    
    documents_context = "\n".join(formatted_docs) if formatted_docs else "No references provided."
    
    # Define the user content block.
    user_content = f"""**References:**
{documents_context}

**Question:**
{question}"""
    
    return user_content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--size', type=int, default=5000)
    parser.add_argument('--log-every', type=int, default=200, help='Print a status line every N records (0 to disable)')
    parser.add_argument('--verbose', action='store_true', help='Print per-record detailed status')

    args = parser.parse_args()

    # data_source = 'ASQA'
    if args.type == 'train':
        TRAIN_SIZE = args.size
        args.data_path = '/mnt/shared-storage-user/liyafu/runquan/musique/data/musique_full_v1.0_train.jsonl'
    else:
        TEST_SIZE = args.size
        args.data_path = '/mnt/shared-storage-user/liyafu/runquan/musique/data/musique_full_v1.0_dev.jsonl'
    
    args.hdfs_dir = None
    args.local_dir = 'data/musique'

    class QASentenceConverter:
        """Convert (question, answer, prior answers) into a single declarative sentence."""
        
        def __init__(self):
            pass

        def convert_one(self, question: str, answer: str, prior_answers: list) -> str:
            q = str(question or '').strip()
            a = str(answer or '').strip()
            if not q or not a:
                return ''
            # Simple template without API calls
            return f"{a} is the answer to: {q}."

    # 读取 JSONL 并在内存中构建列表
    def load_examples_to_list(path, sample_size=None, seed=42):
        # 首先读取所有记录的基本信息
        all_records = []
        with open(path) as f:
            for line in f:
                all_records.append(json.loads(line))
        
        # 如果需要抽样，在处理evidences前就进行抽样
        if sample_size is not None and sample_size < len(all_records):
            random.seed(seed)
            selected_indices = random.sample(range(len(all_records)), sample_size)
            selected_records = [all_records[i] for i in selected_indices]
        else:
            selected_records = all_records
        
        print(f"Total records: {len(all_records)}, Selected: {len(selected_records)}")
        
        # 只处理选中的记录
        qa_converter = QASentenceConverter()
        log_every = args.log_every
        verbose = args.verbose
        records = []
        for idx_data, data in enumerate(tqdm(selected_records)):
            # Ensure data_source field exists
            if 'data_source' not in data:
                data['data_source'] = 'musique'
            if 'paragraphs' in data:
                paragraphs = data['paragraphs'] if isinstance(data['paragraphs'], list) else []
                documents = []
                for para in paragraphs:
                    title = para.get('title', '')
                    paragraph_text = para.get('paragraph_text', '')
                    documents.append([title, [paragraph_text]])
                data['documents'] = str(documents)
                del data['paragraphs']

            # Build evidences from question_decomposition ONLY for train split
            if args.type == 'train':
                evidences_list = []
                qd = data.get('question_decomposition')
                if isinstance(qd, list) and qd:
                    prior_answers = []
                    for qa_idx, qa in enumerate(qd):
                        q = (qa or {}).get('question', '')
                        a = (qa or {}).get('answer', '')
                        sent = qa_converter.convert_one(q, a, prior_answers)
                        if sent:
                            evidences_list.append(sent)
                        # Record current answer for future #n resolution
                        if isinstance(a, str) and a.strip():
                            prior_answers.append(a.strip())
                    # Remove the original decomposition as it's now summarized
                    del data['question_decomposition']

                # Save evidences as string representation for consistency
                data['evidences'] = str(evidences_list)

                # Logging
                if verbose or (log_every > 0 and (idx_data + 1) % log_every == 0):
                    print(
                        f"[musique] idx={idx_data + 1}/{len(selected_records)} "
                        f"evidences={len(evidences_list)}"
                    )
            else:
                # For test split, do not generate evidences; ensure empty and drop qd if present
                if 'question_decomposition' in data:
                    del data['question_decomposition']
                data['evidences'] = '[]'
            
            records.append(data)
        return records
    
    # 直接在加载时进行抽样
    sample_size = TRAIN_SIZE if args.type == 'train' else TEST_SIZE
    raw_records = load_examples_to_list(args.data_path, sample_size=sample_size, seed=42)
    raw_dataset = Dataset.from_list(raw_records)
    print("raw_dataset length: ", len(raw_dataset))

    # 不需要再抽样，直接使用已经抽样好的数据
    if args.type == 'train':
        train_dataset = raw_dataset
        print("train_dataset length: ", len(train_dataset))
    else:
        test_dataset = raw_dataset
        print("test_dataset length: ", len(test_dataset))

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = make_prefix(example)
            data_source = example.get('data_source', 'musique')
            answer = example.get('answer', '')
            answer_aliases = example.get('answer_aliases', [])
            sample_id = example.get('id')
            answerable = True
            evidences = example.get('evidences', '[]')
            original_question = example.get('question', '')
            data = {
                "prompt": prompt,
                "question": original_question,
                "answer": answer,
                "data_source": data_source,
                "evidences": evidences,
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'sample_id': sample_id,
                    'answer_aliases': answer_aliases,
                    'answerable': True,
                }
            }
            return data

        return process_fn


    if args.type == 'train':
        # Remove original columns after they've been formatted into prompt
        # Keep 'documents' and 'evidences' for reference
        cols_to_remove = [col for col in train_dataset.column_names if col not in ['prompt', 'answer', 'data_source', 'extra_info', 'documents']]
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=cols_to_remove)
    else:
        # Remove original columns after they've been formatted into prompt
        # Keep 'documents' and 'evidences' for reference
        cols_to_remove = [col for col in test_dataset.column_names if col not in ['prompt', 'answer', 'data_source', 'extra_info', 'documents']]
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=cols_to_remove)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)


    if args.type == 'train':
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    else:
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

# python -m examples.data_preprocess.musique_full