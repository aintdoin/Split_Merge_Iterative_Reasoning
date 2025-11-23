import pandas as pd
import sys
from transformers import AutoTokenizer
import os

def find_max_prompt_length(file_paths, model_name="gpt2"):
    """
    Reads a list of Parquet files and finds the maximum token length of the 'prompt' field.

    Args:
        file_paths (list): A list of strings, where each string is a path to a Parquet file.
        model_name (str): Name of the model to use for tokenization. Defaults to "gpt2".

    Returns:
        int: The maximum token length of a prompt found across all files.
             Returns 0 if no files or prompts are found.
    """
    max_len = 0
    if not file_paths:
        print("No file paths provided.")
        return 0
    
    # Load tokenizer
    try:
        print(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Using GPT2 tokenizer as fallback.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    for file_path in file_paths:
        try:
            print(f"Processing file: {file_path}")
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            if 'prompt' not in df.columns:
                print(f"Warning: 'prompt' column not found in {file_path}. Skipping.")
                continue

            # Ensure prompt column is of string type, handling potential non-string data
            prompts = df['prompt'].astype(str)
            
            # Calculate token length for each prompt
            token_lens = []
            for prompt in prompts:
                try:
                    tokens = tokenizer(prompt, truncation=False)['input_ids']
                    token_lens.append(len(tokens))
                except Exception as e:
                    print(f"Error tokenizing prompt: {e}")
                    token_lens.append(0)
            
            if token_lens:
                current_max = max(token_lens)
                print(f"Max prompt token length in this file: {current_max}")
                if current_max > max_len:
                    max_len = current_max

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    return max_len

if __name__ == "__main__":
    # From main_grpo_deepseek_distill_qwen.sh
    files_to_check = [
        '/mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/data/musique_truthrl/test_true.parquet',
        '/mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/data/NQ_truthrl/test_true.parquet',
        '/mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/data/hotpot_truthrl/test_true.parquet',
        '/mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/data/CRAG_truthrl/test_true.parquet'
    ]
    
    # Default model for tokenization
    model_name = "gpt2"
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        # Check if the first argument is a model name (contains '/')
        if '/' in sys.argv[1] or os.path.exists(sys.argv[1]):
            model_name = sys.argv[1]
            files_to_check = sys.argv[2:] if len(sys.argv) > 2 else files_to_check
        else:
            files_to_check = sys.argv[1:]

    print(f"Checking files: {files_to_check}")
    print(f"Using model for tokenization: {model_name}")
    overall_max_length = find_max_prompt_length(files_to_check, model_name)

    if overall_max_length > 0:
        print(f"\nOverall maximum prompt token length found: {overall_max_length}")
    else:
        print("\nCould not determine the maximum prompt token length.")
