import json
import os
from datasets import Dataset


def make_prefix(dp):
    """Generate prompt from question and documents"""
    question = dp.get('question', 'no question')
    documents_str = dp.get('documents', '[]')
    
    # Parse and format the documents
    import ast
    try:
        documents_list = ast.literal_eval(documents_str)
    except Exception:
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
    
    # Construct user content with references and question
    user_content = f"""**References:**
{documents_context}

**Question:**
{question}"""
    
    return user_content


def make_map_fn(split, name):
    """Create a mapping function for dataset processing"""
    def process_fn(example, idx):
        query = example.get('query', '')
        retrieved_chunks = example.get('retrieved_chunks', [])
        interaction_id = example.get('interaction_id', '')
        answer = example.get('answer', '')
        alt_ans = example.get('alt_ans', [])
        
        # Create documents from retrieved_chunks
        documents = []
        for chunk in retrieved_chunks:
            page_id = chunk.get('page_id', 0)
            chunk_text = chunk.get('chunk_text', '')
            # Use page_id as title for consistency
            title = f"Page {page_id}"
            documents.append([title, [chunk_text]])
        documents_str = str(documents)
        
        # Create data point for make_prefix
        dp = {
            'question': query,
            'documents': documents_str
        }
        prompt = make_prefix(dp)
        
        # Build the final data structure
        data = {
            "prompt": prompt,
            "question": query,
            "answer": answer[0],
            "data_source": f'truthrl_{name.lower()}',
            "evidences": '[]',
            "extra_info": {
                'split': split,
                'index': idx,
                'sample_id': interaction_id,
                'answer_aliases': alt_ans,
                'answerable': True,
            }
        }
        return data
    
    return process_fn


def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process TruthRL datasets')
    parser.add_argument('--name', choices=['hotpot', 'CRAG', 'NQ', 'musique'], 
                        required=True, help='Dataset name')
    args = parser.parse_args()
    
    # Configuration based on input name
    input_file = f'/mnt/shared-storage-user/liyafu/runquan/TruthRL/downloaded_datasets/{args.name}_test.json'
    output_dir = f'data/{args.name}_truthrl'
    split = 'test_true'
    
    # Read the JSONL file
    print(f"Reading input file: {input_file}")
    raw_records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                raw_records.append(sample)
    
    print(f"Loaded {len(raw_records)} records")
    
    # Create dataset from raw records
    raw_dataset = Dataset.from_list(raw_records)
    
    # Process the dataset
    print("Processing dataset...")
    processed_dataset = raw_dataset.map(make_map_fn(split, args.name), with_indices=True, remove_columns=raw_dataset.column_names)  # 移除原始列
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to parquet file
    output_file = os.path.join(output_dir, f'{split}.parquet')
    processed_dataset.to_parquet(output_file)
    
    print(f"Processing complete. Saved {len(processed_dataset)} examples to {output_file}")


if __name__ == '__main__':
    main()