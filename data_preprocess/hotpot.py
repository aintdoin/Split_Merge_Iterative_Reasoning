""" 
合并hotpot数据处理和Best-of-N过滤的完整pipeline
1. 先处理answerable样本（size/2条）
2. 生成unanswerable样本并使用Best-of-32过滤，直到获得足够的样本（size/2条）
3. 混合并保存
"""

import os
import sys

# CRITICAL: Must set this before ANY imports that might use torch/CUDA
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
from datasets import Dataset
import ast
import random

# Add parent directory to path to import verl modules
# File is in data_preprocess/, so we need to go up one level to reach project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional imports - requests for API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("警告: requests不可用，无法使用API模式")


def make_prefix_unified(dp, template_type):
    """统一的prompt前缀，用于answerable和unanswerable样本"""
    question = dp.get('question', 'no question')
    documents_str = dp.get('documents', '[]')
    
    # 解析并格式化documents
    try:
        documents_list = ast.literal_eval(documents_str)
    except:
        documents_list = []
    
    formatted_docs = []
    for doc in documents_list:
        if isinstance(doc, list) and len(doc) == 2:
            title, sentences = doc
            if isinstance(sentences, list):
                text = ' '.join(str(s) for s in sentences)
            else:
                text = str(sentences)
            formatted_docs.append(f"Document '{title}': {text}")
    
    documents_context = "\n".join(formatted_docs) if formatted_docs else "No references provided."
    
    user_content = f"""**References:**
{documents_context}

**Question:**
{question}"""
    
    system_prompt = """You are a helpful assistant. You are given a Question and References.

Your task: answer the Question only using factual information contained in the References. Do not use any external knowledge or your own knowledge.

**CRITICAL - You MUST follow this EXACT format:**
<think>
1. [First reasoning step]
2. [Second reasoning step]
3. [Third reasoning step]
...
</think>
<answer>Your final answer</answer>

**Rules (STRICTLY ENFORCED):**
1. Put reasoning in <think></think> tags
2. Use numbered steps (1., 2., 3., ...) in your <think> section for clear structured reasoning
3. NEVER start with anything other than <think> or <answer>
4. The <answer> tag MUST contain your final answer

Remember: Any response without proper <answer></answer> tags is INCORRECT."""
    
    if template_type in ['qwen']:
        prefix = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>"""
    elif template_type in ['llama']:
        prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let me solve this step by step.
<think>"""
    else:
        prefix = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>"""
    return prefix


def gen_from_jsonl(path):
    """从JSONL文件加载数据并转换为dataset格式"""
    with open(path) as f:
        for line in f:
            data = json.loads(line)    
            if 'supporting_facts' in data:
                evidence = []
                for fact in data['supporting_facts']:
                    title, sent_idx = fact
                    for doc in data['context']:
                        if doc[0] == title:
                            doc_text = " ".join(doc[1])
                            evidence.append(doc_text)
                            break
                data['evidences'] = str(evidence)
                data['supporting_facts'] = str(data['supporting_facts'])
            if 'context' in data:
                data['documents'] = str(data['context'])
                del data['context']


            if '_id' in data:
                extra_info = data.get('extra_info', {})
                if not isinstance(extra_info, dict):
                    extra_info = {}
                extra_info['sample_id'] = str(data['_id'])
                data['extra_info'] = extra_info
            yield data


def _has_valid_format(text: str) -> bool:
    """检查文本是否有有效的<answer></answer>格式"""
    try:
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


def _is_idk_answer(text: str) -> bool:
    """检查答案是否是IDK/不确定的表达"""
    if not text:
        return False
    text_lower = text.strip().lower()
    idk_markers = [
        "i don't know", "i dont know", "i do not know",
        "i'm not sure", "i am not sure", "not sure",
        "cannot answer", "can't answer", "unable to answer",
        "cannot determine", "can't determine", "unable to determine",
        "insufficient information", "not enough information",
        "no sufficient information", "lack of information",
        "unknown", "unclear", "uncertain",
    ]
    return any(marker in text_lower for marker in idk_markers)


def call_api_for_candidates(prompt: str, api_base: str, model_name: str, api_key: str,
                            n: int, temperature: float, top_p: float, top_k: int, max_tokens: int) -> list:
    """
    调用API生成N个候选回答
    返回生成的文本列表
    """
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests库不可用，无法使用API模式")
    
    base = api_base.rstrip('/')
    chat_url = base + '/v1/chat/completions'
    comp_url = base + '/v1/completions'
    
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    # 尝试chat endpoint
    chat_payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'n': n,
        'stream': False,
    }
    
    resp = requests.post(chat_url, json=chat_payload, headers=headers, timeout=120)
    if resp.status_code == 200:
        data = resp.json()
        candidates = []
        try:
            for choice in data.get('choices', []):
                text = choice.get('message', {}).get('content', '')
                if text and text.strip():
                    candidates.append(text.strip())
        except Exception:
            pass
        if candidates:
            return candidates
    
    # 备用：completions endpoint
    comp_payload = {
        'model': model_name,
        'prompt': prompt,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'n': n,
        'stream': False,
    }
    
    resp2 = requests.post(comp_url, json=comp_payload, headers=headers, timeout=120)
    if resp2.status_code != 200:
        print(f"\n⚠️  API错误详情:")
        print(f"  状态: {resp2.status_code}")
        print(f"  URL: {comp_url}")
        print(f"  模型: {model_name}")
        print(f"  响应: {resp2.text[:300]}")
        return []
    
    data2 = resp2.json()
    candidates = []
    for choice in data2.get('choices', []):
        text = choice.get('text', '').strip()
        if text:
            candidates.append(text)
    
    return candidates if candidates else []


def evaluate_sample_best_of_n(sample_dict, prompt, args, llm, sampling_params, postprocessor):
    """
    使用Best-of-N策略评估单个样本
    返回: (is_truly_unanswerable: bool, best_reward: float)
    
    如果32次推理中有任何一次成功回答（非IDK且正确），则返回False（不是真正的unanswerable）
    只有32次全部失败，才返回True（是真正的unanswerable）
    """
    import re
    
    # 提取元数据
    extra_info = sample_dict.get('extra_info', {})
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except:
            extra_info = {}
    
    question = sample_dict.get('question', '')
    ground_truth = sample_dict.get('answer', '')
    answer_aliases = extra_info.get('answer_aliases', [])
    if isinstance(answer_aliases, np.ndarray):
        answer_aliases = answer_aliases.tolist()
    elif answer_aliases is None:
        answer_aliases = []
    
    # 生成N个候选回答
    if args.use_api:
        # API模式
        candidates = call_api_for_candidates(
            prompt, args.api_base, args.model_name, args.api_key,
            args.n_candidates, args.temperature, args.top_p, args.top_k, args.max_tokens
        )
        
        if not candidates:
            print(f"  ⚠️ API调用失败，跳过此样本")
            return False, -2
        
        # 评估所有候选
        has_correct_answer = False
        best_reward = -1
        
        for candidate_text in candidates:
            # 检查格式
            if not _has_valid_format(candidate_text):
                reward = -1
            else:
                # 提取答案
                match = re.search(r'<answer>\s*(.*?)\s*</answer>', candidate_text, re.DOTALL | re.IGNORECASE)
                extracted_answer = match.group(1).strip() if match else candidate_text.strip()
                
                # 检查是否是IDK
                is_idk = _is_idk_answer(extracted_answer)
                
                # 尝试所有可能的答案
                reward_scores = []
                all_answers = [ground_truth]
                if answer_aliases and len(answer_aliases) > 0:
                    all_answers.extend(answer_aliases)
                
                for ans in all_answers:
                    if ans:
                        try:
                            score = postprocessor.judge_answer_correctness(
                                predicted_answer=candidate_text,
                                ground_truth_answer=ans,
                                question=question,
                                answerable=False
                            )
                            reward_scores.append(score)
                        except Exception:
                            continue
                
                reward = max(reward_scores) if reward_scores else 0
                
                # 只有非IDK且正确才算成功回答
                if reward >= 0.999 and not is_idk:
                    has_correct_answer = True
            
            if reward > best_reward:
                best_reward = reward
        
        # 返回结果：如果有正确答案，则不是真正的unanswerable
        return not has_correct_answer, best_reward
        
    else:
        # 本地vLLM模式
        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        has_correct_answer = False
        best_reward = -1
        
        for candidate_output in output.outputs:
            generated_text = candidate_output.text
            
            # 检查格式
            if not _has_valid_format(generated_text):
                reward = -1
            else:
                # 提取答案
                match = re.search(r'<answer>\s*(.*?)\s*</answer>', generated_text, re.DOTALL | re.IGNORECASE)
                extracted_answer = match.group(1).strip() if match else generated_text.strip()
                
                # 检查是否是IDK
                is_idk = _is_idk_answer(extracted_answer)
                
                # 尝试所有可能的答案
                reward_scores = []
                all_answers = [ground_truth]
                if answer_aliases and len(answer_aliases) > 0:
                    all_answers.extend(answer_aliases)
                
                for ans in all_answers:
                    if ans:
                        try:
                            score = postprocessor.judge_answer_correctness(
                                predicted_answer=generated_text,
                                ground_truth_answer=ans,
                                question=question,
                                answerable=False
                            )
                            reward_scores.append(score)
                        except Exception:
                            continue
                
                reward = max(reward_scores) if reward_scores else 0
                
                # 只有非IDK且正确才算成功回答
                if reward >= 0.999 and not is_idk:
                    has_correct_answer = True
            
            if reward > best_reward:
                best_reward = reward
        
        return not has_correct_answer, best_reward


def create_answerable_samples(dataset, num_samples, template_type):
    """创建answerable样本"""
    print(f"\n{'='*80}")
    print(f"步骤1: 创建{num_samples}条ANSWERABLE样本")
    print(f"{'='*80}")
    
    answerable_dataset = dataset.select(range(num_samples))
    print(f"选取了{len(answerable_dataset)}个样本用于answerable")
    
    answerable_samples = []
    for i in range(len(answerable_dataset)):
        sample = answerable_dataset[i]
        answerable_sample = {
            'question': sample.get('question', ''),
            'documents': sample.get('documents', '[]'),
            'answer': sample['answer'],
            'data_source': sample.get('data_source', 'hotpot'),
            'evidences': sample.get('evidences', '[]'),
            'extra_info': sample.get('extra_info', {}),
        }
        
        # 设置answerable为True
        if isinstance(answerable_sample['extra_info'], dict):
            answerable_sample['extra_info']['answerable'] = True
        else:
            answerable_sample['extra_info'] = {'answerable': True}
        
        answerable_samples.append(answerable_sample)
    
    print(f"✓ 创建了{len(answerable_samples)}条answerable样本")
    return answerable_samples


def create_unanswerable_samples_with_filter(dataset, start_idx, num_samples, template_type, 
                                           args, llm, sampling_params, postprocessor):
    """
    创建unanswerable样本并实时过滤
    边生成边过滤，直到获得足够的样本
    """
    print(f"\n{'='*80}")
    print(f"步骤2: 创建并过滤{num_samples}条UNANSWERABLE样本")
    print(f"{'='*80}")
    print(f"使用Best-of-{args.n_candidates}策略")
    print(f"只保留32次推理全部失败的样本\n")
    
    kept_samples = []
    removed_count = 0
    processed_count = 0
    
    # 我们需要两倍的样本：一部分作为问题，另一部分作为不匹配的文档
    max_samples_to_try = min(len(dataset) - start_idx, num_samples * 10)  # 最多尝试10倍数量
    
    pbar = tqdm(total=num_samples, desc="过滤unanswerable样本")
    
    idx = start_idx
    while len(kept_samples) < num_samples and idx < len(dataset) - 1:
        question_sample = dataset[idx]
        
        # 获取问题和原始文档
        question = question_sample.get('question', '')
        answer = question_sample['answer']
        
        # 解析原始文档
        try:
            original_documents = ast.literal_eval(question_sample.get('documents', '[]'))
        except:
            original_documents = []
        
        # 解析supporting_facts以识别关键文档
        supporting_facts = question_sample.get('supporting_facts', None)
        if supporting_facts is None:
            extra_info = question_sample.get('extra_info', {})
            if isinstance(extra_info, str):
                try:
                    extra_info = ast.literal_eval(extra_info)
                except:
                    extra_info = {}
            supporting_facts = extra_info.get('supporting_facts', [])
        
        if isinstance(supporting_facts, str):
            try:
                supporting_facts = ast.literal_eval(supporting_facts)
            except:
                supporting_facts = []
        
        # 提取唯一的文档标题
        supporting_doc_titles = []
        if isinstance(supporting_facts, list):
            for fact in supporting_facts:
                if isinstance(fact, list) and len(fact) >= 1:
                    title = fact[0]
                    if title not in supporting_doc_titles:
                        supporting_doc_titles.append(title)
        
        # 策略：移除关键支撑文档（除了起始节点）
        if supporting_doc_titles and len(supporting_doc_titles) > 1 and original_documents:
            starting_node = supporting_doc_titles[0]
            removal_candidates = supporting_doc_titles[1:]
            
            if len(supporting_doc_titles) >= 4:
                num_to_remove = min(2, len(removal_candidates))
            else:
                num_to_remove = 1
            
            docs_to_remove = random.sample(removal_candidates, num_to_remove)
            
            # 移除选定的文档
            modified_documents = original_documents.copy()
            for doc_title in docs_to_remove:
                modified_documents = [doc for doc in modified_documents 
                                    if not (isinstance(doc, list) and len(doc) >= 2 and doc[0] == doc_title)]
        else:
            modified_documents = original_documents
        
        # 增强evidences
        evidences = question_sample.get('evidences', '[]')
        try:
            evidences_list = ast.literal_eval(evidences) if isinstance(evidences, str) else evidences
        except:
            evidences_list = []
        
        augmented_evidences = evidences_list
        
        # 创建unanswerable样本
        unanswerable_sample = {
            'question': question,
            'documents': str(modified_documents),
            'answer': answer,
            'data_source': question_sample.get('data_source', 'hotpot'),
            'evidences': str(augmented_evidences),
            'extra_info': question_sample.get('extra_info', {}),
        }
        
        # 更新answerable为False（不修改或创建 sample_id）
        if isinstance(unanswerable_sample['extra_info'], dict):
            unanswerable_sample['extra_info']['answerable'] = False
        else:
            unanswerable_sample['extra_info'] = {'answerable': False}
        
        # 生成prompt用于过滤
        prompt = make_prefix_unified(unanswerable_sample, template_type)
        
        # 使用Best-of-N评估
        is_truly_unanswerable, best_reward = evaluate_sample_best_of_n(
            unanswerable_sample, prompt, args, llm, sampling_params, postprocessor
        )
        
        processed_count += 1
        
        if is_truly_unanswerable:
            # 通过过滤：32次全部失败
            kept_samples.append(unanswerable_sample)
            pbar.update(1)
            pbar.set_postfix({
                '保留': len(kept_samples),
                '移除': removed_count,
                '处理': processed_count,
                '移除率': f'{removed_count/processed_count*100:.1f}%'
            })
        else:
            # 未通过过滤：至少有一次成功回答
            removed_count += 1
        
        idx += 1
        
        if idx >= len(dataset) - 1:
            print(f"\n⚠️ 警告: 已遍历完所有可用样本")
            print(f"   只获得了{len(kept_samples)}/{num_samples}条合格的unanswerable样本")
            break
    
    pbar.close()
    
    print(f"\n{'='*80}")
    print("过滤结果")
    print(f"{'='*80}")
    print(f"处理的样本总数: {processed_count}")
    print(f"保留的样本(真正unanswerable): {len(kept_samples)}")
    print(f"移除的样本(可被回答): {removed_count}")
    print(f"移除率: {removed_count/processed_count*100:.1f}%")
    
    return kept_samples


def main():
    parser = argparse.ArgumentParser(description='合并hotpot数据处理和Best-of-N过滤')
    parser.add_argument('--type', type=str, default='train', help='train或test')
    parser.add_argument('--template_type', type=str, default='deepseek-r1-distill-qwen')
    parser.add_argument('--size', type=int, required=True, help='目标样本总数（将平分为answerable和unanswerable）')
    
    # 数据路径
    parser.add_argument('--data-path', type=str, default=None, help='输入JSONL文件路径')
    
    # 模型/API配置
    parser.add_argument('--model-path', type=str, default='', help='本地模型路径（vLLM模式）')
    parser.add_argument('--use-api', action='store_true', help='使用API模式而非本地vLLM')
    parser.add_argument('--api-base', type=str, default='http://localhost:8000', help='API基础URL')
    parser.add_argument('--api-key', type=str, default='', help='API密钥（可选）')
    parser.add_argument('--model-name', type=str, default='', help='API模型名称')
    
    # 过滤参数
    parser.add_argument('--n-candidates', type=int, default=32, help='每个样本生成的候选回答数量')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p采样参数')
    parser.add_argument('--top-k', type=int, default=100, help='Top-k采样参数')
    parser.add_argument('--max-tokens', type=int, default=2048, help='最大生成token数')
    
    # vLLM参数
    parser.add_argument('--max-model-len', type=int, default=24500, help='vLLM最大模型长度')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='vLLM张量并行大小')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("HOTPOT数据处理 + BEST-OF-N过滤 合并PIPELINE")
    print("="*80)
    print(f"总目标样本数: {args.size}")
    print(f"  - Answerable: {args.size // 2}")
    print(f"  - Unanswerable (需过滤): {args.size // 2}")
    print(f"模式: {'API' if args.use_api else '本地vLLM'}")
    if args.use_api:
        print(f"API Base: {args.api_base}")
        print(f"模型: {args.model_name}")
    else:
        print(f"模型路径: {args.model_path}")
    print(f"Best-of-N: {args.n_candidates}")
    print("="*80)
    
    # 确定数据路径
    if args.data_path:
        data_path = args.data_path
    elif args.type == 'train':
        data_path = '/mnt/shared-storage-user/liyafu/runquan/hotpot/hotpot_train_v1.1.jsonl'
    else:
        data_path = '/mnt/shared-storage-user/liyafu/runquan/hotpot/hotpot_dev_distractor_v1.jsonl'
    
    # 计算每类样本数量
    answerable_size = args.size // 2
    unanswerable_size = args.size // 2
    
    # 加载原始数据集
    print(f"\n📂 从{data_path}加载数据...")
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': data_path})
    print(f"   ✓ 原始数据集长度: {len(raw_dataset)}")
    
    # 打乱并选择足够的样本
    total_needed = answerable_size + unanswerable_size * 10  # 为unanswerable预留更多样本
    dataset = raw_dataset.shuffle(seed=42).select(range(min(total_needed, len(raw_dataset))))
    print(f"   ✓ 选择了{len(dataset)}个样本用于处理")
    
    # 初始化模型/API
    llm = None
    sampling_params = None
    
    if args.use_api:
        print(f"\n🌐 测试API连接...")
        if not REQUESTS_AVAILABLE:
            print("错误: requests库不可用!")
            return
        if not args.model_name:
            print("错误: API模式需要--model-name参数!")
            return
        
        base = args.api_base.rstrip('/')
        models_url = base + '/v1/models'
        try:
            resp = requests.get(models_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                available_models = [m.get('id', 'unknown') for m in data.get('data', [])]
                print(f"  ✓ API可访问")
                print(f"  可用模型: {available_models}")
                if args.model_name not in available_models and available_models:
                    print(f"  ⚠️ 警告: '{args.model_name}'不在可用模型列表中")
            else:
                print(f"  ⚠️ 警告: 无法访问models端点 (状态: {resp.status_code})")
        except Exception as e:
            print(f"  ⚠️ 警告: 无法连接到API: {e}")
            return
    else:
        print(f"\n🔧 从{args.model_path}加载模型...")
        if not args.model_path:
            print("错误: 本地模式需要--model-path参数!")
            return
        
        # 延迟导入vLLM，只在实际使用时导入
        try:
            from vllm import LLM, SamplingParams
            print("   ✓ vLLM模块导入成功")
        except ImportError:
            print("错误: vLLM不可用! 请使用--use-api切换到API模式。")
            return
        
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len
        )
        
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            n=args.n_candidates,
        )
        print("   ✓ 模型加载完成")
    
    # 初始化answer postprocessor (在vLLM加载之后导入，避免CUDA冲突)
    print("\n🔍 初始化answer postprocessor...")
    from verl.utils.reward_score.answer_postprocessor import get_postprocessor
    postprocessor = get_postprocessor()
    print("   ✓ Postprocessor初始化完成")
    
    # 步骤1: 创建answerable样本
    answerable_samples = create_answerable_samples(dataset, answerable_size, args.template_type)
    
    # 步骤2: 创建并过滤unanswerable样本
    unanswerable_samples = create_unanswerable_samples_with_filter(
        dataset, answerable_size, unanswerable_size, args.template_type,
        args, llm, sampling_params, postprocessor
    )
    
    # 合并样本
    print(f"\n{'='*80}")
    print("步骤3: 合并并保存数据集")
    print(f"{'='*80}")
    
    all_samples = answerable_samples + unanswerable_samples
    print(f"总样本数: {len(all_samples)}")
    print(f"  - Answerable: {len(answerable_samples)}")
    print(f"  - Unanswerable: {len(unanswerable_samples)}")
    
    # 转换为Dataset
    combined_dataset = Dataset.from_list(all_samples)
    
    # 重新生成prompt
    def regenerate_prompt(example, idx):
        question = make_prefix_unified(example, template_type=args.template_type)
        return {
            "prompt": question,
            "question": example['question'],
            "answer": example['answer'],
            "data_source": example['data_source'],
            "extra_info": example['extra_info'],
            "documents": example['documents'],
            "evidences": example['evidences'],
        }
    
    print("\n生成prompt...")
    combined_dataset = combined_dataset.map(regenerate_prompt, with_indices=True)
    
    # 洗混数据
    print("洗混数据集...")
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    # 保存
    output_dir = f'data/hotpot/{args.template_type}'
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    
    if args.type == 'train':
        output_file = os.path.join(output_dir, 'train.parquet')
    else:
        output_file = os.path.join(output_dir, 'test.parquet')
    
    combined_dataset.to_parquet(output_file)
    print(f"\n💾 保存到{output_file}")
    print(f"   ✓ 最终数据集: {len(combined_dataset)}个样本")
    
    # 验证
    df_verify = pd.read_parquet(output_file)
    n_false = sum(1 for _, row in df_verify.iterrows() 
                  if isinstance(row.get('extra_info'), (dict, str)) and 
                  (json.loads(row['extra_info']) if isinstance(row['extra_info'], str) else row['extra_info']).get('answerable') == False)
    n_true = sum(1 for _, row in df_verify.iterrows() 
                 if isinstance(row.get('extra_info'), (dict, str)) and 
                 (json.loads(row['extra_info']) if isinstance(row['extra_info'], str) else row['extra_info']).get('answerable') == True)
    
    print(f"\n验证:")
    print(f"   ✓ answerable=True: {n_true}")
    print(f"   ✓ answerable=False: {n_false}")
    
    # 清理
    if llm is not None:
        try:
            llm.shutdown()
        except Exception:
            pass
    
    print("\n✅ 完成!\n")


if __name__ == '__main__':
    main()

