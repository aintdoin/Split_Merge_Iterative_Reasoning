#!/usr/bin/env python3
"""
Extract samples from test.parquet and run inference with confidence filtering.
Supports filtering by answerable status, multiple datasets and models.
"""

import pandas as pd
import json
from vllm import LLM, SamplingParams
import os
import sys
from tqdm import tqdm
import numpy as np
import argparse

# Add parent directory to path to import verl modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verl.utils.reward_score.answer_postprocessor import get_postprocessor


def series_to_item(ls):
    """
    Unwrap pandas Series or numpy arrays to get the actual data.
    This is the same function used in training code.
    """
    import pandas, numpy
    while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
        ls = ls[0]
    return ls

def convert_to_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run inference on samples from test.parquet with confidence filtering"
    )
    parser.add_argument(
        "--test-files",
        type=str,
        required=True,
        help="Test data files in Python list format, e.g., \"['data/hotpot/qwen/test.parquet', 'data/musique/qwen/test.parquet']\""
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output file path for inference results (complete path including filename, e.g., 'inference/results/output.jsonl')"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name for file organization (e.g., deepseek-r1-qwen-7b, qwen-7b)"
    )
    parser.add_argument(
        "--filter-type",
        type=str,
        choices=["all", "answerable", "unanswerable"],
        default="all",
        help="Filter samples by answerable status (default: all)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=40000,
        help="Number of samples to process (default: 40000, use -1 for all matching samples)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=24500,
        help="Maximum model length for vLLM (default: 24500)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default: 1)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (default: 1.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Optional: Path to verl checkpoint (e.g., 'checkpoints/shuffle/qwen_7b/grpo/global_step_100'). Will be merged to HuggingFace format if needed."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=-float('inf'),
        help="Threshold for confidence score (cumulative logprob). Answers with lower confidence will be replaced with 'I don't know'."
    )
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Configuration based on arguments
    # Handle checkpoint loading if specified
    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    print(f"Checkpoint path argument: '{args.checkpoint_path}'")
    print(f"Checkpoint path is empty: {not args.checkpoint_path}")
    print(f"Checkpoint path type: {type(args.checkpoint_path)}")
    
    if args.checkpoint_path and args.checkpoint_path.strip():
        print(f"\n→ Checkpoint loading enabled")
        checkpoint_path = args.checkpoint_path.strip()
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)
        
        print(f"  Original path: {args.checkpoint_path}")
        print(f"  Absolute path: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"\n✗ Error: Checkpoint path does not exist: {checkpoint_path}")
            print(f"  Falling back to base model: {args.model_path}")
            MODEL_PATH = args.model_path
        else:
            print(f"  ✓ Checkpoint path exists")
            
            # Check for actor subdirectory (verl checkpoint structure)
            actor_path = os.path.join(checkpoint_path, 'actor')
            if os.path.exists(actor_path):
                checkpoint_to_merge = actor_path
                print(f"  ✓ Found 'actor' subdirectory")
            else:
                checkpoint_to_merge = checkpoint_path
                print(f"  ⚠ No 'actor' subdirectory, using checkpoint path directly")
            
            # Check if already merged
            merged_path = os.path.join(checkpoint_path, 'merged_hf')
            merged_config = os.path.join(merged_path, 'config.json')
            
            if os.path.exists(merged_config):
                print(f"\n  ✓ Found existing merged checkpoint!")
                print(f"    Path: {merged_path}")
                MODEL_PATH = merged_path
            else:
                # Need to merge FSDP shards
                print(f"\n  → Merging FSDP checkpoint to HuggingFace format...")
                print(f"    Source: {checkpoint_to_merge}")
                print(f"    Target: {merged_path}")
                
                import subprocess
                merge_script = os.path.join(os.path.dirname(__file__), 'merge_fsdp_checkpoint.py')
                
                print(f"    Merge script: {merge_script}")
                print(f"    Script exists: {os.path.exists(merge_script)}")
                print(f"\n  → Starting merge process...")
                
                result = subprocess.run([
                    sys.executable,
                    merge_script,
                    '--checkpoint', checkpoint_to_merge,
                    '--output', merged_path,
                    '--base-model', args.model_path
                ])
                
                print(f"\n  → Merge process completed with return code: {result.returncode}")
                print(f"    Checking for merged config: {merged_config}")
                print(f"    Config exists: {os.path.exists(merged_config)}")
                
                if result.returncode == 0 and os.path.exists(merged_config):
                    print(f"  ✓ Merge successful!")
                    MODEL_PATH = merged_path
                else:
                    print(f"  ✗ Merge failed, falling back to base model")
                    MODEL_PATH = args.model_path
            
            print(f"\n→ Final model path: {MODEL_PATH}")
    else:
        print(f"\n→ No checkpoint specified, using base model")
        MODEL_PATH = args.model_path
        print(f"  Base model path: {args.model_path}")
    
    print("="*80 + "\n")
    
    # Parse test files (convert string representation of list to actual list)
    import ast
    try:
        test_files = ast.literal_eval(args.test_files)
        if not isinstance(test_files, list):
            test_files = [test_files]
    except:
        # If parsing fails, treat as single file path
        test_files = [args.test_files]
    
    # Use the output path directly as specified by user
    OUTPUT_FILE = args.output_dir
    
    # Create parent directory if needed
    output_parent = os.path.dirname(OUTPUT_FILE)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    NUM_SAMPLES = args.num_samples
    
    print(f"Loading data from {len(test_files)} file(s):")
    for f in test_files:
        print(f"  - {f}")
    print(f"Filter type: {args.filter_type}")
    print(f"Confidence Threshold: {args.confidence_threshold}")
    
    # Read and concatenate all parquet files
    dfs = []
    for file_path in test_files:
        if os.path.exists(file_path):
            df_single = pd.read_parquet(file_path)
            print(f"  Loaded {len(df_single)} samples from {file_path}")
            dfs.append(df_single)
        else:
            print(f"  Warning: File not found: {file_path}")
    
    if not dfs:
        raise ValueError("No valid data files found!")
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal samples in dataset: {len(df)}")
    
    # Filter samples based on answerable status
    filtered_samples = []
    for idx, row in df.iterrows():
        extra_info = row['extra_info']
        # Handle different formats of extra_info
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        
        if isinstance(extra_info, dict):
            answerable = extra_info.get('answerable')
            
            # Apply filter based on filter_type
            if args.filter_type == "all":
                filtered_samples.append(row)
            elif args.filter_type == "answerable":
                # Check for True
                if answerable is True or answerable == True:
                    filtered_samples.append(row)
            elif args.filter_type == "unanswerable":
                # Check for False
                if answerable is False or answerable == False:
                    filtered_samples.append(row)
    
    print(f"Found {len(filtered_samples)} {args.filter_type} samples")
    
    # Select NUM_SAMPLES samples (or all if NUM_SAMPLES == -1)
    if NUM_SAMPLES == -1 or len(filtered_samples) <= NUM_SAMPLES:
        if NUM_SAMPLES != -1 and len(filtered_samples) < NUM_SAMPLES:
            print(f"Warning: Only {len(filtered_samples)} samples available, using all of them")
        selected_samples = filtered_samples
    else:
        selected_samples = filtered_samples[:NUM_SAMPLES]
    
    print(f"Selected {len(selected_samples)} samples for inference")
    
    # Load the model
    print(f"Loading model from {MODEL_PATH}...")
    print(f"Model configuration:")
    print(f"  - Max model length: {args.max_model_len}")
    print(f"  - Tensor parallel size: {args.tensor_parallel_size}")
    
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",  # Match veRL validation (ppo_trainer.yaml line 72)
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len
    )
    
    # Sampling parameters
    print(f"Sampling configuration:")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Top-k: {args.top_k}")
    print(f"  - Repetition penalty: {args.repetition_penalty}")
    print(f"  - Max tokens: {args.max_tokens}")
    print(f"  - Logprobs: 1 (for confidence calculation)")
    
    # Align with validation: when temperature==0, force greedy settings
    sp_top_p = args.top_p
    sp_top_k = args.top_k
    if args.temperature == 0:
        sp_top_p = 1.0
        sp_top_k = -1
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=sp_top_p,
        top_k=sp_top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        logprobs=1  # Request logprobs for confidence calculation
    )
    
    # Prepare prompts for inference
    prompts = []
    
    # Debug: Check first sample structure
    if selected_samples:
        first_sample = selected_samples[0]
        if isinstance(first_sample, pd.Series):
            first_dict = first_sample.to_dict()
        else:
            first_dict = first_sample
    
    for idx, sample in enumerate(selected_samples):
        # Convert sample to dict if it's a pandas Series
        if isinstance(sample, pd.Series):
            sample_dict = sample.to_dict()
        else:
            sample_dict = sample
            
        # Extract the prompt text from the sample
        # IMPORTANT: Use series_to_item to unwrap pandas/numpy wrappers
        prompt_raw = sample_dict.get('prompt', [])
        prompt = series_to_item(prompt_raw)
        
        # Handle different prompt formats
        prompt_text = ''
        
        # Case 1: prompt is directly a dict (after unwrapping)
        if isinstance(prompt, dict):
            prompt_text = prompt.get('content', '')
        # Case 2: prompt is a list
        elif isinstance(prompt, list) and len(prompt) > 0:
            first_elem = prompt[0]
            if isinstance(first_elem, dict):
                prompt_text = first_elem.get('content', '')
            elif isinstance(first_elem, str):
                prompt_text = first_elem
        # Case 3: prompt is a string
        elif isinstance(prompt, str):
            # Might be JSON string, try to parse it
            try:
                prompt_parsed = json.loads(prompt)
                if isinstance(prompt_parsed, list) and len(prompt_parsed) > 0:
                    prompt_text = prompt_parsed[0].get('content', '') if isinstance(prompt_parsed[0], dict) else ''
                elif isinstance(prompt_parsed, dict):
                    prompt_text = prompt_parsed.get('content', '')
                else:
                    prompt_text = prompt  # Use as-is
            except:
                prompt_text = prompt  # Use as-is if not JSON
        
        # Debug: show first 3 samples
        if idx < 3:
            print(f"\n[Sample {idx}] Prompt extracted successfully")
            print(f"  Type after unwrap: {type(prompt)}")
            print(f"  Length: {len(prompt_text)} chars")
            print(f"  Preview: {prompt_text[:150]}...")
        
        if not prompt_text or not prompt_text.strip():
            print(f"\n[WARNING] Sample {idx} has empty prompt!")
            print(f"  Prompt type: {type(prompt)}")
            print(f"  Prompt value: {prompt}")
        
        prompts.append(prompt_text)
    
    print(f"\nRunning inference on {len(prompts)} prompts...")
    
    # Check for empty prompts
    empty_count = sum(1 for p in prompts if not p or not p.strip())
    if empty_count > 0:
        # Show indices of empty prompts
        empty_indices = [i for i, p in enumerate(prompts) if not p or not p.strip()]
        raise ValueError(f"Found {empty_count} empty prompts in the data")

    # System prompt injection (默认启用，使用统一的 system_prompts 模块)
    enable_injection = os.environ.get('ENABLE_SYSTEM_PROMPT_INJECTION', 'true').lower() == 'true'
    
    if enable_injection:
        from verl.utils.dataset.system_prompts import wrap_prompt_with_system
        
        # 默认使用 qwen 模板，可通过环境变量覆盖
        model_template = os.environ.get('MODEL_TEMPLATE', 'qwen')
        
        print(f"\n🔄 启用运行时 System Prompt 注入")
        print(f"   模板类型: {model_template}")
        
        # 对所有 prompts 应用 system prompt 注入
        # wrap_prompt_with_system 会自动检测并移除旧的 system prompt（如果存在）
        prompts = [wrap_prompt_with_system(p, model_template=model_template) for p in prompts]
    # Show first prompt preview
    if prompts:
        print(f"\nFirst prompt preview:")
        print(f"  {prompts[0]}...")
    
    # Run inference
    outputs = llm.generate(prompts, sampling_params)
    # Initialize postprocessor for reward evaluation
    print("\nInitializing answer postprocessor for reward evaluation...")
    postprocessor = get_postprocessor()
    
    # Prepare results
    print(f"\nEvaluating rewards for {len(outputs)} samples...")
    results = []
    # Helper: extract content inside <answer>...</answer>
    def _extract_answer_content(text: str) -> str:
        if not text:
            return ""
        try:
            ps = text.lower().find('<answer>')
            pe = text.lower().find('</answer>')
            if ps != -1 and pe != -1 and ps < pe:
                return text[ps + len('<answer>'):pe].strip()
        except Exception:
            pass
        return text.strip()
    # Helper: normalize for case-insensitive exact match
    def _normalize(s: str) -> str:
        return str(s or "").strip().lower()
    # Helper: IDK detection aligned with postprocessor
    _IDK_MARKERS = {"i don't know", "i dont know", "insufficient information", "unknown"}
    def _is_idk(s: str) -> bool:
        return _normalize(s) in _IDK_MARKERS
        
    # Counter for filtered samples
    filtered_count = 0
    
    for i, (sample, output) in enumerate(tqdm(zip(selected_samples, outputs), total=len(outputs), desc="Evaluating rewards")):
        generated_text = output.outputs[0].text
        cumulative_logprob = output.outputs[0].cumulative_logprob
        
        # Confidence filtering logic
        is_filtered = False
        original_text = generated_text
        
        # Check if confidence is below threshold
        # Note: cumulative_logprob is a float (sum of logprobs)
        if cumulative_logprob < args.confidence_threshold:
            generated_text = "<answer>I don't know</answer>"
            is_filtered = True
            filtered_count += 1
        
        # Convert sample to dict if it's a pandas Series
        if isinstance(sample, pd.Series):
            sample_dict = sample.to_dict()
        else:
            sample_dict = sample
        
        # Extract extra_info
        extra_info = sample_dict.get('extra_info', {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        
        # Extract prompt content (full prompt with documents)
        # Use series_to_item to unwrap pandas/numpy wrappers
        prompt_raw = sample_dict.get('prompt', [])
        prompt = series_to_item(prompt_raw)
        
        # Handle different prompt formats
        if isinstance(prompt, dict):
            # Prompt is directly a dict after unwrapping
            prompt_content = prompt.get('content', '')
        elif isinstance(prompt, list) and len(prompt) > 0:
            first_elem = prompt[0]
            if isinstance(first_elem, dict):
                prompt_content = first_elem.get('content', '')
            elif isinstance(first_elem, str):
                prompt_content = first_elem
            else:
                prompt_content = ''
        else:
            prompt_content = ''
        
        # Extract question (from top-level sample_dict, not extra_info)
        question = sample_dict.get('question', '')
        
        # Extract ground truth and other fields
        ground_truth = sample_dict.get('answer', '')
        answer_aliases_raw = extra_info.get('answer_aliases', [])
        # Convert to list if it's a numpy array
        if isinstance(answer_aliases_raw, np.ndarray):
            answer_aliases = answer_aliases_raw.tolist()
        elif answer_aliases_raw is None:
            answer_aliases = []
        else:
            answer_aliases = answer_aliases_raw
        
        answerable = extra_info.get('answerable', True)
        
        # Evaluate reward using postprocessor
        # First, strict format check: missing tags or wrong order -> reward = -1
        def _has_valid_format(text: str) -> bool:
            try:
                # Require exactly one <answer>...</answer> pair with non-empty content
                a_s = text.count('<answer>')
                a_e = text.count('</answer>')
                if a_s != 1 or a_e != 1:
                    return False
                ps = text.find('<answer>')
                pe = text.find('</answer>')
                if ps == -1 or pe == -1 or ps >= pe:
                    return False
                content = text[ps + len('<answer>'):pe].strip()
                if len(content) == 0:
                    return False
                return True
            except Exception:
                return False

        if not _has_valid_format(generated_text):
            reward = -1
        else:
            # Extract final answer once
            predicted_final = _extract_answer_content(generated_text)
            # Fast path 1: exact match against GT or any alias (case-insensitive)
            all_answers = [ground_truth]
            if answer_aliases and len(answer_aliases) > 0:
                all_answers.extend(answer_aliases)
            normalized_pred = _normalize(predicted_final)
            normalized_set = {_normalize(a) for a in all_answers if isinstance(a, str) and a}
            if normalized_pred in normalized_set:
                reward = 1
            # Fast path 2: IDK handling without calling judge
            elif _is_idk(predicted_final):
                if answerable is True:
                    reward = 0
                elif answerable is False:
                    reward = 1
                else:
                    reward = 0
            else:
                # Judge path: try GT first, then aliases with early-stop on 1
                reward = -1  # default if judge not available or returns -1
                try_order = [ground_truth] + [a for a in answer_aliases if isinstance(a, str) and a]
                for ans in try_order:
                    try:
                        score = postprocessor.judge_answer_correctness(
                            predicted_answer=predicted_final,
                            ground_truth_answer=ans,
                            question=question,
                            answerable=answerable
                        )
                        if score == 1:
                            reward = 1
                            break
                        # keep reward as -1 if not 1
                    except Exception as e:
                        # On any failure, continue trying other aliases
                        continue
        
        result = {
            'sample_id': extra_info.get('sample_id', f'sample_{i}'),
            'index': extra_info.get('index', i),
            'question': question,  # Pure question text
            'question_prompt': prompt_content,  # Full prompt with documents
            'ground_truth_answer': ground_truth,
            'answer_aliases': answer_aliases,
            'answerable': answerable,
            'data_source': sample_dict.get('data_source', ''),
            'model_output': generated_text,
            'original_output': original_text, # Store original output for reference
            'confidence': cumulative_logprob, # Store confidence score
            'is_filtered': is_filtered, # Flag if filtered
            'reward': reward,  # Evaluated reward: +1, 0, or -1
        }
        
        # Convert all numpy types to JSON-serializable types
        result = convert_to_serializable(result)
        results.append(result)
    
    # Save results to JSONL file
    print(f"Saving results to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Successfully saved {len(results)} inference results to {OUTPUT_FILE}")
    
    # Print a summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model: {args.model_name}")
    if args.checkpoint_path:
        print(f"Checkpoint: {args.checkpoint_path}")
    else:
        print(f"Base Model: {args.model_path}")
    print(f"Total samples processed: {len(results)}")
    print(f"Filtered samples (Low Confidence): {filtered_count} ({filtered_count/len(results)*100:.2f}%)")
    print(f"Input files ({len(test_files)} file(s)):")
    for f in test_files:
        print(f"  - {f}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Print first result as example
    if results:
        print("\nExample result (first sample):")
        print(f"Sample ID: {results[0]['sample_id']}")
        print(f"Question: {results[0]['question'][:150]}..." if len(results[0]['question']) > 150 else f"Question: {results[0]['question']}")
        print(f"Answerable: {results[0]['answerable']}")
        print(f"Ground Truth Answer: {results[0]['ground_truth_answer']}")
        print(f"Model Output (first 200 chars): {results[0]['model_output'][:200]}...")
        print(f"Confidence: {results[0]['confidence']}")
        print(f"Is Filtered: {results[0]['is_filtered']}")
        print(f"Reward: {results[0]['reward']}")

    # Also report three-class test_score to match training validation
    rewards = [r.get('reward', 0) for r in results]
    n = len(rewards)
    if n > 0:
        n_correct = sum(1 for v in rewards if v >= 0.999)
        n_miss = sum(1 for v in rewards if -0.001 < v < 0.001)
        # Treat others (including -1 and -2) as incorrect
        n_incorrect = n - n_correct - n_miss
        test_score = 2.0 * (n_correct / n) + (n_miss / n) - 1.0
        avg_reward = sum(rewards) / n

        print("\n" + "-"*80)
        print("Aggregate metrics (aligned with training test_score):")
        print(f"  n = {n}")
        print(f"  n_correct = {n_correct} ({n_correct/n:.3f})")
        print(f"  n_miss    = {n_miss} ({n_miss/n:.3f})")
        print(f"  n_incorrect = {n_incorrect} ({n_incorrect/n:.3f})")
        print(f"  test_score = {test_score:.6f}")
        print(f"  avg_reward (includes -2 for bad format) = {avg_reward:.6f}")
    
    # Cleanup
    try:
        llm.shutdown()
    except Exception:
        pass

if __name__ == '__main__':
    main()

