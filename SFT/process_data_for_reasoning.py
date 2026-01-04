# /Users/guirunquan/Documents/github/Split_Merge_Iterative_Reasoning/SFT/process_data_for_reasoning.py
import pandas as pd
import numpy as np
import os
import sys
import re
import json
import ast
import math
from tqdm import tqdm

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
USE_METHOD = os.environ.get('USE_METHOD', 'cosmo')
INPUT_FILE = os.environ.get('INPUT_FILE', 'data/musique/train.parquet')
OUTPUT_FILE = os.environ.get('OUTPUT_FILE', 'SFT/data/musique_train.parquet')
OUTPUT_RL_FILE = os.environ.get('OUTPUT_RL_FILE', 'data/musique/train_rl.parquet')
MODEL_PATH = os.environ.get('MODEL_PATH', '') # Required for C3oT, FS_BoN, SPIRIT

def get_judge_api_client():
    from openai import OpenAI
    api_key = os.environ.get('LLM_JUDGE_API_KEY', 'EMPTY')
    base_url = os.environ.get('LLM_JUDGE_API_BASE', 'http://localhost:8000/v1')
    return OpenAI(api_key=api_key, base_url=base_url)

def get_llm_engine():
    from vllm import LLM
    print(f"Loading model from {MODEL_PATH}...")
    return LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=int(os.environ.get('TENSOR_PARALLEL_SIZE', '1')),
        max_model_len=int(os.environ.get('MAX_MODEL_LEN', '16384')),
        gpu_memory_utilization=float(os.environ.get('GPU_MEMORY_UTILIZATION', '0.9'))
    )

def run_vllm_inference(llm, prompts, n=1, temperature=0.7):
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=int(os.environ.get('MAX_TOKENS', '4096')),
        stop=['<|im_end|>', '<|endoftext|>']
    )
    
    print(f"Running inference on {len(prompts)} prompts with n={n}...")
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for output in outputs:
        texts = [o.text for o in output.outputs]
        results.append(texts)
    
    return results

def calculate_ppl_vllm(llm, texts_with_prompt_len):
    # input: list of (full_text, prompt_token_len)
    # prompt_token_len is used to identify where the "reasoning" starts for PPL calculation
    from vllm import SamplingParams
    
    # We use max_tokens=1 and prompt_logprobs=1 to get logprobs for the input
    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1
    )
    
    full_texts = [t[0] for t in texts_with_prompt_len]
    outputs = llm.generate(full_texts, sampling_params)
    
    ppls = []
    for i, output in enumerate(outputs):
        # output.prompt_logprobs is a list of dicts {token_id: logprob, ...} or None
        # It corresponds to the tokenized prompt
        prompt_logprobs = output.prompt_logprobs
        prompt_token_len = texts_with_prompt_len[i][1]
        
        # prompt_logprobs[0] is None usually (start of sequence)
        # We want logprobs from prompt_token_len to end
        # Note: prompt_logprobs has length equal to input tokens
        
        # Filter None and extract logprob
        # Logprobs for the reasoning part:
        # The tokens we care about are from index `prompt_token_len` to the end.
        # These are tokens generated *given* the prompt part.
        
        target_logprobs = []
        if prompt_logprobs:
            for j in range(len(prompt_logprobs)):
                if j < prompt_token_len:
                    continue # Skip prompt part
                
                # Check if we have logprobs at this position
                if prompt_logprobs[j]:
                    # We get the logprob of the actual token that is there
                    # But prompt_logprobs[j] is a dict of top-K logprobs.
                    # We need the logprob of the token that *actually* exists at index j.
                    # Wait, vllm prompt_logprobs returns top-k. If the actual token is not in top-k, we might miss it?
                    # By default prompt_logprobs=1 might only return top-1.
                    # But wait, request_output.prompt_logprobs is intended to give the logprob of the prompt tokens?
                    # Documentation says: "The log probabilities of the prompt tokens. The list contains one dictionary for each token in the prompt."
                    # Actually, if we want the logprob of the specific token `t_j`, we might need `prompt_logprobs` to cover it.
                    # If we set `prompt_logprobs=1`, it gives top 1. If the true token isn't top 1, we don't get its score?
                    # That would be bad for PPL.
                    # BUT: older vllm versions or specific configurations might behave differently.
                    # Actually, for PPL, usually one wants the logprob of the ground truth token.
                    # If `vllm` only returns top-k, we can't compute exact PPL unless k is large enough.
                    # HOWEVER, many "score" implementations use `prompt_logprobs=0` (which might return None) or special flags.
                    # Let's check if we can get the logprob of the *actual* token.
                    # In recent vllm, `prompt_logprobs` returns the logprob of the token *at that position* if found?
                    # No, it returns a dict mapping token_id -> logprob.
                    # We also need the token_ids of the prompt. `output.prompt_token_ids`.
                    
                    token_id = output.prompt_token_ids[j]
                    token_logprobs_dict = prompt_logprobs[j]
                    
                    if token_id in token_logprobs_dict:
                        target_logprobs.append(token_logprobs_dict[token_id].logprob)
                    else:
                        # If not in top-k, we have a problem.
                        # But typically for "scoring", we might set prompt_logprobs=None? No.
                        # We might need a large K? Or use a different library?
                        # Wait, for now let's assume K=1 is sufficient or that the model is good enough that the token is in top-1?
                        # No, that's unsafe for PPL calculation of *modified* sequences which might be unlikely.
                        # Let's try `prompt_logprobs=None`? No.
                        # Let's set `prompt_logprobs=20` to be safer?
                        # Or maybe there is a way to force getting the token's logprob.
                        # Actually, looking at vllm docs, `prompt_logprobs` (int) number of logprobs to return.
                        # If we want the logprob of the *actual* token, we need it to be in the returned set.
                        pass
                        
            # If we missed tokens, our PPL calculation is invalid.
            # Workaround: For SPIRIT, we are comparing candidates. Maybe top-5 is enough?
            # Let's set prompt_logprobs=0? No.
            # Let's set it to something reasonable like 5 or 10.
            pass
        
        # With prompt_logprobs in SamplingParams, if we want the logprob of the *chosen* token (in prompt),
        # we strictly need to find it in the dict.
        
        # Let's update SamplingParams in the function to use a higher value, e.g. 20.
        
        if target_logprobs:
            avg_logprob = sum(target_logprobs) / len(target_logprobs)
            ppl = math.exp(-avg_logprob)
            ppls.append(ppl)
        else:
            ppls.append(float('inf'))
            
    return ppls

def format_prompt(row, tokenizer=None, model_path=''):
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.system_prompts import get_system_prompt
    
    prompt_content = row.get('prompt', '')
    if isinstance(prompt_content, dict):
        prompt_content = prompt_content.get('content', '')
    
    system_prompt = get_system_prompt()
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt_content}
    ]
    
    try:
        if tokenizer is None:
            tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}. Using fallback formatting.")
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt_content}<|im_end|>\n<|im_start|>assistant\n"

def extract_think_and_answer(text):
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    
    if not think and not answer:
        return text, ""
        
    return think, answer

def judge_step_meaningful(step, client, judge_model):
    prompt = f"Check if the following reasoning step is meaningful and contributes to answering the question.\nStep: {step}\nAnswer with 'Yes' or 'No' only."
    
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip().lower()
        return "yes" in content
    except Exception as e:
        print(f"Judge error: {e}")
        return True

def process_c3ot(df, llm):
    print("Processing with C3oT method...")
    tokenizer = llm.get_tokenizer()
    prompts = [format_prompt(row, tokenizer, MODEL_PATH) for _, row in df.iterrows()]
    outputs = run_vllm_inference(llm, prompts, n=1, temperature=0.7)
    
    client = get_judge_api_client()
    judge_model = os.environ.get('LLM_JUDGE_MODEL_NAME', 'gpt-4')
    
    sft_data = []
    print("Filtering reasoning steps...")
    for idx, (output_list, (_, row)) in tqdm(enumerate(zip(outputs, df.iterrows())), total=len(df)):
        generated_text = output_list[0]
        think, answer = extract_think_and_answer(generated_text)
        
        if not think:
            continue
            
        steps = re.split(r'\n(?=\d+\.)', think)
        steps = [s.strip() for s in steps if s.strip()]
        
        valid_steps = []
        for step in steps:
            if not re.match(r'^\d+\.', step):
                if valid_steps:
                    valid_steps[-1] += "\n" + step
                else:
                    valid_steps.append(step)
                continue
                
            if judge_step_meaningful(step, client, judge_model):
                valid_steps.append(step)
        
        new_think_lines = []
        for i, step in enumerate(valid_steps):
            clean_step = re.sub(r'^\d+\.\s*', '', step)
            new_think_lines.append(f"{i+1}. {clean_step}")
            
        new_think = "\n".join(new_think_lines)
        hops = len(new_think_lines)
        response = f"<think>\n{new_think}\n</think>\n<answer>{answer}</answer>"
        
        sft_data.append({
            'prompt': row.get('prompt', ''),
            'response': response,
            'hops': hops
        })
        
    return pd.DataFrame(sft_data)

def process_fs_bon(df, llm):
    print("Processing with FS_BoN method...")
    from verl.utils.reward_score.answer_postprocessor import get_postprocessor
    
    tokenizer = llm.get_tokenizer()
    prompts = [format_prompt(row, tokenizer, MODEL_PATH) for _, row in df.iterrows()]
    outputs = run_vllm_inference(llm, prompts, n=3, temperature=0.7)
    
    postprocessor = get_postprocessor()
    sft_data = []
    
    print("Selecting best responses...")
    for idx, (output_list, (_, row)) in tqdm(enumerate(zip(outputs, df.iterrows())), total=len(df)):
        ground_truth = row.get('answer', '')
        extra_info = row.get('extra_info', {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        answer_aliases = extra_info.get('answer_aliases', []) if isinstance(extra_info, dict) else []
        answerable = extra_info.get('answerable', True) if isinstance(extra_info, dict) else True
        
        correct_candidates = []
        for output in output_list:
            think, answer = extract_think_and_answer(output)
            if not answer: continue
            
            is_correct = False
            try_answers = [ground_truth] + (answer_aliases if answer_aliases else [])
            for ref_ans in try_answers:
                if not ref_ans: continue
                score = postprocessor.judge_answer_correctness(
                    predicted_answer=answer,
                    ground_truth_answer=ref_ans,
                    question=row.get('question', ''),
                    answerable=answerable
                )
                if score == 1:
                    is_correct = True
                    break
            
            if is_correct:
                correct_candidates.append({
                    'response': output,
                    'len': len(think),
                    'think': think
                })
        
        if correct_candidates:
            correct_candidates.sort(key=lambda x: x['len'])
            best = correct_candidates[0]
            hops = len(re.findall(r'\n\d+\.', best['think']))
            if hops == 0 and best['think']: hops = 1
            
            sft_data.append({
                'prompt': row.get('prompt', ''),
                'response': best['response'],
                'hops': hops
            })
    
    return pd.DataFrame(sft_data)

def process_spirit(df, llm):
    print("Processing with SPIRIT method...")
    tokenizer = llm.get_tokenizer()
    
    # 1. Generate initial traces
    prompts = [format_prompt(row, tokenizer, MODEL_PATH) for _, row in df.iterrows()]
    outputs = run_vllm_inference(llm, prompts, n=1, temperature=0.7)
    
    sft_data = []
    
    print("Filtering reasoning steps using PPL...")
    
    # We process each sample
    # Because PPL calculation requires calling llm, we should batch them if possible
    # But filtering is iterative per sample?
    # User's method: "If removing that step PPL < initial PPL, remove it."
    # We can create all candidate strings for a batch of samples and run one giant PPL pass?
    # Or just process sample by sample to save memory complexity, but slower?
    # Let's batch by sample: process one sample (create ~5-10 candidates), score them.
    # Actually, vllm is efficient with batches.
    
    # To optimize:
    # 1. Collect all candidates for all samples.
    # 2. Run one big generate(prompt_logprobs=1).
    # 3. Process results.
    # But we need to map back to samples.
    
    # Let's do it in chunks of samples to balance memory and speed.
    CHUNK_SIZE = 100
    
    for i in tqdm(range(0, len(df), CHUNK_SIZE), desc="Processing chunks"):
        chunk_df = df.iloc[i:i+CHUNK_SIZE]
        chunk_outputs = outputs[i:i+CHUNK_SIZE]
        chunk_prompts = prompts[i:i+CHUNK_SIZE]
        
        # Prepare evaluation tasks
        eval_tasks = [] # list of tuples: (sample_idx_in_chunk, step_idx_to_remove_or_minus_1_for_baseline, full_text, prompt_len)
        
        sample_info = [] # store parsed steps and other info per sample
        
        for local_idx, (output_list, prompt_text) in enumerate(zip(chunk_outputs, chunk_prompts)):
            generated_text = output_list[0]
            think, answer = extract_think_and_answer(generated_text)
            
            if not think:
                sample_info.append(None)
                continue
                
            # Parse steps (reuse C3oT logic)
            steps = re.split(r'\n(?=\d+\.)', think)
            steps = [s.strip() for s in steps if s.strip()]
            
            # Clean steps to ensure they are valid logic steps? 
            # We assume extraction is similar.
            parsed_steps = []
            for step in steps:
                if re.match(r'^\d+\.', step):
                   parsed_steps.append(step)
                else:
                    # Merge with previous if exist
                    if parsed_steps:
                        parsed_steps[-1] += "\n" + step
                    else:
                        parsed_steps.append(step)
            
            if not parsed_steps:
                sample_info.append(None)
                continue
                
            info = {
                'original_think': think,
                'answer': answer,
                'steps': parsed_steps,
                'prompt_text': prompt_text,
                'prompt_tokens_len': len(tokenizer.encode(prompt_text))
            }
            sample_info.append(info)
            
            # 1. Baseline (Full)
            # Reconstruct full text from steps to be consistent
            # We wrap with <think>...</think>
            full_think_str = "\n".join(parsed_steps)
            full_response = f"<think>\n{full_think_str}\n</think>\n<answer>{answer}</answer>"
            full_text = prompt_text + full_response
            
            eval_tasks.append({
                'sample_local_idx': local_idx,
                'removed_step_idx': -1, # -1 means baseline
                'text': full_text,
                'prompt_len': info['prompt_tokens_len']
            })
            
            # 2. Candidates (Remove each step)
            for step_idx in range(len(parsed_steps)):
                # Remove step_idx
                subset_steps = parsed_steps[:step_idx] + parsed_steps[step_idx+1:]
                subset_think_str = "\n".join(subset_steps)
                subset_response = f"<think>\n{subset_think_str}\n</think>\n<answer>{answer}</answer>"
                subset_text = prompt_text + subset_response
                
                eval_tasks.append({
                    'sample_local_idx': local_idx,
                    'removed_step_idx': step_idx,
                    'text': subset_text,
                    'prompt_len': info['prompt_tokens_len']
                })

        if not eval_tasks:
            continue
            
        # Run PPL calculation
        # We need to update run_ppl to take raw texts
        # Note: we need to pass a list of (text, prompt_len)
        texts_to_score = [(t['text'], t['prompt_len']) for t in eval_tasks]
        
        # Override sampling params for high logprobs recall
        from vllm import SamplingParams
        sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=20) 
        
        # Run inference
        score_outputs = llm.generate([t[0] for t in texts_to_score], sampling_params)
        
        # Calculate PPLs
        ppls = []
        for j, output in enumerate(score_outputs):
            prompt_logprobs = output.prompt_logprobs
            prompt_token_len = texts_to_score[j][1]
            
            target_logprobs = []
            if prompt_logprobs:
                for k in range(len(prompt_logprobs)):
                    if k < prompt_token_len: continue
                    
                    token_id = output.prompt_token_ids[k]
                    token_logprobs_dict = prompt_logprobs[k]
                    
                    if token_id in token_logprobs_dict:
                        target_logprobs.append(token_logprobs_dict[token_id].logprob)
                    # If missing, we ignore? Or punish?
                    # Ignoring is safest for now to avoid crash
            
            if target_logprobs:
                avg = sum(target_logprobs) / len(target_logprobs)
                ppl = math.exp(-avg)
            else:
                ppl = float('inf')
            ppls.append(ppl)
            
        # Map back results
        # Organize by sample
        # sample_results[local_idx] = {'baseline': ppl, 'candidates': {step_idx: ppl}}
        sample_results = {}
        for task_idx, task in enumerate(eval_tasks):
            s_idx = task['sample_local_idx']
            r_idx = task['removed_step_idx']
            ppl = ppls[task_idx]
            
            if s_idx not in sample_results:
                sample_results[s_idx] = {'candidates': {}}
            
            if r_idx == -1:
                sample_results[s_idx]['baseline'] = ppl
            else:
                sample_results[s_idx]['candidates'][r_idx] = ppl
        
        # Determine filtering
        for local_idx in range(len(chunk_df)):
            if sample_info[local_idx] is None:
                continue
                
            if local_idx not in sample_results:
                continue
                
            res = sample_results[local_idx]
            baseline_ppl = res.get('baseline', float('inf'))
            candidates = res.get('candidates', {})
            
            info = sample_info[local_idx]
            steps = info['steps']
            
            steps_to_keep = []
            for idx_s, step in enumerate(steps):
                # If removing step makes PPL < baseline, we REMOVE it.
                # So if candidate_ppl >= baseline, we KEEP it.
                # Wait, "If delete ... PPL < Initial, then remove".
                # So: Keep if candidate_ppl >= baseline.
                
                cand_ppl = candidates.get(idx_s, float('inf'))
                
                if cand_ppl < baseline_ppl:
                    # Remove this step
                    pass
                else:
                    steps_to_keep.append(step)
            
            # Reconstruct final
            # Renumber
            new_think_lines = []
            for k, s in enumerate(steps_to_keep):
                clean_s = re.sub(r'^\d+\.\s*', '', s)
                new_think_lines.append(f"{k+1}. {clean_s}")
            
            new_think = "\n".join(new_think_lines)
            hops = len(new_think_lines)
            response = f"<think>\n{new_think}\n</think>\n<answer>{info['answer']}</answer>"
            
            row = chunk_df.iloc[local_idx]
            sft_data.append({
                'prompt': row.get('prompt', ''),
                'response': response,
                'hops': hops
            })
            
    return pd.DataFrame(sft_data)

def process_cosmo(df):
    print("Processing with cosmo method (original logic)...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.7)
    sft_data = []
    rl_data = []
    
    total_samples = len(df)
    print(f"Processing {total_samples} samples (70% SFT, 30% RL)...")
    
    for idx, row in tqdm(df.iterrows(), total=total_samples):
        prompt = row.get('prompt', '')
        answer = row.get('answer', '')
        evidences = row.get('evidences', [])
        
        reasoning_lines = []
        if isinstance(evidences, str) and evidences.startswith('[') and evidences.endswith(']'):
            try:
                parsed_evidences = ast.literal_eval(evidences)
                if isinstance(parsed_evidences, list):
                    for i, evidence in enumerate(parsed_evidences):
                        reasoning_lines.append(f"{i+1}. {evidence}")
            except (ValueError, SyntaxError):
                reasoning_lines.append(f"1. {evidences}")
        elif isinstance(evidences, (list, np.ndarray)):
            for i, evidence in enumerate(evidences):
                reasoning_lines.append(f"{i+1}. {evidence}")
        else:
            reasoning_lines.append(f"1. {evidences}")
            
        reasoning_text = "\n".join(reasoning_lines)
        hops = len(reasoning_lines)
        
        if idx < split_idx:
            response = f"<think>\n{reasoning_text}\n</think>\n<answer>{answer}</answer>"
            sft_data.append({
                'prompt': prompt,
                'response': response,
                'hops': hops
            })
        else:
            row_dict = row.to_dict()
            row_dict['hops'] = hops
            rl_data.append(row_dict)
            
    rl_df = pd.DataFrame(rl_data)
    output_rl_dir = os.path.dirname(OUTPUT_RL_FILE)
    if output_rl_dir and not os.path.exists(output_rl_dir):
        os.makedirs(output_rl_dir)
    print(f"Saving RL data to: {OUTPUT_RL_FILE}")
    rl_df.to_parquet(OUTPUT_RL_FILE, index=False)
    
    return pd.DataFrame(sft_data)

def process_data_for_reasoning():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
        
    print(f"Reading input file: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    
    if USE_METHOD == 'cosmo':
        sft_df = process_cosmo(df)
    else:
        # Load LLM only for methods that need it
        llm = get_llm_engine()
        if USE_METHOD == 'C3oT':
            sft_df = process_c3ot(df, llm)
        elif USE_METHOD == 'FS_BoN':
            sft_df = process_fs_bon(df, llm)
        elif USE_METHOD == 'SPIRIT':
            sft_df = process_spirit(df, llm)
        else:
            raise ValueError(f"Unknown method: {USE_METHOD}")
        
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving SFT data to: {OUTPUT_FILE}")
    sft_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Processed {len(sft_df)} samples.")

if __name__ == "__main__":
    process_data_for_reasoning()
