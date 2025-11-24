""" 
处理2WikiMultihop数据，将三元组evidences通过LLM API转化为自然语言句子，并生成answerable=true的样本。

流程:
1) 针对原始数据，按 format 流程转换到目标格式，直接将 evidences 三元组用 LLM API 转换为句子
2) 赋予 answerable=True，保存到输出文件
"""

import os
import sys

# CRITICAL: Must set this before ANY imports that might use torch/CUDA
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import ast
import json
import argparse
import random
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# Add parent directory to path to import verl modules
# File is in data_preprocess/, so we need to go up one level to reach project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =========================
# Prompt & dataset helpers
# =========================

def make_prefix_unified(dp: dict, template_type: str) -> str:
    """统一的prompt前缀，用于answerable和unanswerable样本"""
    question = dp.get('question', 'no question')
    documents_str = dp.get('documents', '[]')
    
    # 解析并格式化documents
    try:
        documents_list = ast.literal_eval(documents_str) if isinstance(documents_str, str) else documents_str
    except Exception:
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
    
    return user_content


def gen_from_jsonl(path: str):
    """从JSONL文件加载数据并转换为dataset格式（2WikiMultihop版本）"""
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            # 2wikimultihop: 若有supporting_facts，保持字符串形式便于后续解析
            if 'supporting_facts' in data:
                data['supporting_facts'] = str(data['supporting_facts'])
            # evidences可能是三元组列表，这里直接转为字符串保存（稍后再结构化处理）
            if 'evidences' in data:
                try:
                    data['evidences'] = str(data['evidences'])
                except Exception:
                    data['evidences'] = '[]'
            # 统一将context改为documents
            if 'context' in data:
                data['documents'] = str(data['context'])
                del data['context']

            # 简单设置 sample_id
            if '_id' in data:
                extra_info = data.get('extra_info', {})
                if not isinstance(extra_info, dict):
                    extra_info = {}
                extra_info['sample_id'] = str(data['_id'])
                data['extra_info'] = extra_info
            yield data


# =========================
# Evidence conversion via API
# =========================


def to_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            return list(ast.literal_eval(s))
        except Exception:
            try:
                return list(json.loads(s))
            except Exception:
                return []
    return []


def clean_evidence_items(items: List[Any]) -> List[List[Any]]:
    result: List[List[Any]] = []
    for item in items:
        if isinstance(item, list) and len(item) == 3:
            result.append(item)
    return result


class TripleSentenceConverter:
    """
    用于将 (subject, relation, object) 三元组转化为自然语言句子。
    优先调用 Chat Completions API；若不可用则回退为确定性模板。
    
    通过以下环境变量配置:
      LLM_JUDGE_API_BASE, LLM_JUDGE_MODEL_NAME, LLM_JUDGE_API_KEY, LLM_JUDGE_TIMEOUT, LLM_JUDGE_MAX_WORKERS
    """
    def __init__(self):
        self.api_base = os.environ.get('LLM_JUDGE_API_BASE', '').strip()
        self.model_name = os.environ.get('LLM_JUDGE_MODEL_NAME', '').strip() or 'llm-judge'
        self.api_key = os.environ.get('LLM_JUDGE_API_KEY', '').strip()
        try:
            self.timeout = float(os.environ.get('LLM_JUDGE_TIMEOUT', '60'))
        except Exception:
            self.timeout = 60.0
        try:
            self.max_workers = int(os.environ.get('LLM_JUDGE_MAX_WORKERS', '8'))
        except Exception:
            self.max_workers = 8

        self.requests = requests if REQUESTS_AVAILABLE else None
        self.use_api = bool(self.api_base and self.requests)

    def _build_messages(self, subject: str, relation: str, obj: str) -> list:
        system_content = (
            "You are an expert at converting knowledge triples into clear, natural English sentences.\n\n"
            "Task Instructions:\n"
            "1. Transform the triple into ONE grammatically correct sentence\n"
            "2. Maintain the semantic relationship between the subject and object\n"
            "3. Use appropriate phrasing based on the relation type\n"
            "4. Return ONLY the resulting sentence, nothing else\n\n"
            "Example:\n"
            "Triple: ['Stuart Rosenberg', 'director', 'Move (1970 film)']\n"
            "Output: Stuart Rosenberg is the director of Move (1970 film).\n\n"
            "Triple: ['Jean-Daniel Pollet', 'country of citizenship', 'French']\n"
            "Output: Jean-Daniel Pollet's country of citizenship is France."
        )
        user_content = (
            "Convert the following knowledge triple into a single, natural English sentence:\n"
            f"['{subject}', '{relation}', '{obj}']"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _call_chat(self, messages: list) -> str:
        assert self.requests is not None
        base = self.api_base.rstrip('/')
        url = base + '/v1/chat/completions'
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        payload = {
            'model': self.model_name,
            'messages': messages,
            'temperature': 0.0,
            'max_tokens': 80,
            'stream': False,
        }
        resp = self.requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        return text.strip() if text else ''

    def convert_triple(self, triple: List[Any]) -> str:
        subject, relation, obj = (str(triple[0]), str(triple[1]), str(triple[2]))
        if self.use_api:
            try:
                messages = self._build_messages(subject, relation, obj)
                out = self._call_chat(messages)
                if out:
                    return out
            except Exception:
                pass
        return f"{subject} {relation} {obj}."

    def convert_triples(self, triples: List[List[Any]]) -> List[str]:
        if not triples:
            return []
        # Simple sequential for determinism and avoiding too many threads in data preprocess
        return [self.convert_triple(t) for t in triples]


def convert_evidences_to_sentences(evidences_cell: Any, converter: TripleSentenceConverter) -> List[str]:
    items = to_list(evidences_cell)
    triples = clean_evidence_items(items)
    sentences = converter.convert_triples(triples)
    return sentences


# =========================
# Best-of-N filtering utils
# =========================

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
    调用API生成N个候选回答，返回生成的文本列表
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


def evaluate_sample_best_of_n(sample_dict: dict, prompt: str, args, llm, sampling_params, postprocessor):
    """
    使用Best-of-N策略评估单个样本
    返回: (is_truly_unanswerable: bool, best_reward: float)
    
    如果N次推理中有任何一次成功回答（非IDK且正确），则返回False（不是真正的unanswerable）
    只有N次全部失败，才返回True（是真正的unanswerable）
    """
    import re
    
    # 提取元数据
    extra_info = sample_dict.get('extra_info', {})
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except Exception:
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
        candidates = call_api_for_candidates(
            prompt, args.api_base, args.model_name, args.api_key,
            args.n_candidates, args.temperature, args.top_p, args.top_k, args.max_tokens
        )
        
        if not candidates:
            print(f"  ⚠️ API调用失败，跳过此样本")
            return False, -2
        
        has_correct_answer = False
        best_reward = -1
        
        for candidate_text in candidates:
            if not _has_valid_format(candidate_text):
                reward = -1
            else:
                match = re.search(r'<answer>\s*(.*?)\s*</answer>', candidate_text, re.DOTALL | re.IGNORECASE)
                extracted_answer = match.group(1).strip() if match else candidate_text.strip()
                
                is_idk = _is_idk_answer(extracted_answer)
                
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
                
                if reward >= 0.999 and not is_idk:
                    has_correct_answer = True
            
            if reward > best_reward:
                best_reward = reward
        
        return not has_correct_answer, best_reward
        
    else:
        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        has_correct_answer = False
        best_reward = -1
        
        for candidate_output in output.outputs:
            generated_text = candidate_output.text
            
            if not _has_valid_format(generated_text):
                reward = -1
            else:
                match = re.search(r'<answer>\s*(.*?)\s*</answer>', generated_text, re.DOTALL | re.IGNORECASE)
                extracted_answer = match.group(1).strip() if match else generated_text.strip()
                
                is_idk = _is_idk_answer(extracted_answer)
                
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
                
                if reward >= 0.999 and not is_idk:
                    has_correct_answer = True
            
            if reward > best_reward:
                best_reward = reward
        
        return not has_correct_answer, best_reward


# =========================
# Sample builders
# =========================

def create_answerable_samples(dataset: Dataset, num_samples: int, template_type: str,
                              converter: TripleSentenceConverter) -> List[dict]:
    """创建answerable样本，并将 evidences 三元组转为自然语言句子"""
    print(f"\n{'='*80}")
    print(f"步骤1: 创建{num_samples}条ANSWERABLE样本")
    print(f"{'='*80}")
    
    answerable_dataset = dataset.select(range(num_samples))
    print(f"选取了{len(answerable_dataset)}个样本用于answerable")
    
    answerable_samples: List[dict] = []
    for i in range(len(answerable_dataset)):
        sample = answerable_dataset[i]
        evid_sentences = convert_evidences_to_sentences(sample.get('evidences', '[]'), converter)
        answerable_sample = {
            'question': sample.get('question', ''),
            'documents': sample.get('documents', '[]'),
            'answer': sample['answer'],
            'data_source': sample.get('data_source', '2wikimultihop'),
            'evidences': evid_sentences,  # 已转为句子列表
            'extra_info': sample.get('extra_info', {}),
        }
        if isinstance(answerable_sample['extra_info'], dict):
            answerable_sample['extra_info']['answerable'] = True
        else:
            answerable_sample['extra_info'] = {'answerable': True}
        answerable_samples.append(answerable_sample)
    
    print(f"✓ 创建了{len(answerable_samples)}条answerable样本")
    return answerable_samples


def create_unanswerable_samples_with_filter(dataset: Dataset, start_idx: int, num_samples: int,
                                            template_type: str, args, llm, sampling_params,
                                            postprocessor, converter: TripleSentenceConverter) -> List[dict]:
    """
    创建unanswerable样本并实时过滤（Best-of-N）
    不再添加任何IDK句子，仅对documents进行支撑文档移除并过滤
    """
    print(f"\n{'='*80}")
    print(f"步骤2: 创建并过滤{num_samples}条UNANSWERABLE样本")
    print(f"{'='*80}")
    print(f"使用Best-of-{args.n_candidates}策略")
    print(f"只保留{args.n_candidates}次推理全部失败的样本\n")
    
    kept_samples: List[dict] = []
    removed_count = 0
    processed_count = 0
    
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
        except Exception:
            original_documents = []
        
        # 解析supporting_facts以识别关键文档
        supporting_facts = question_sample.get('supporting_facts', None)
        if supporting_facts is None:
            extra_info = question_sample.get('extra_info', {})
            if isinstance(extra_info, str):
                try:
                    extra_info = ast.literal_eval(extra_info)
                except Exception:
                    extra_info = {}
            supporting_facts = extra_info.get('supporting_facts', [])
        
        if isinstance(supporting_facts, str):
            try:
                supporting_facts = ast.literal_eval(supporting_facts)
            except Exception:
                supporting_facts = []
        
        # 提取唯一的文档标题
        supporting_doc_titles: List[str] = []
        if isinstance(supporting_facts, list):
            for fact in supporting_facts:
                if isinstance(fact, list) and len(fact) >= 1:
                    title = fact[0]
                    if title not in supporting_doc_titles:
                        supporting_doc_titles.append(title)
        
        # 策略：移除关键支撑文档（除了起始节点）
        if supporting_doc_titles and len(supporting_doc_titles) > 1 and original_documents:
            removal_candidates = supporting_doc_titles[1:]
            if len(supporting_doc_titles) >= 4:
                num_to_remove = min(2, len(removal_candidates))
            else:
                num_to_remove = 1
            docs_to_remove = random.sample(removal_candidates, num_to_remove)
            modified_documents = original_documents.copy()
            for doc_title in docs_to_remove:
                modified_documents = [doc for doc in modified_documents 
                                      if not (isinstance(doc, list) and len(doc) >= 2 and doc[0] == doc_title)]
        else:
            modified_documents = original_documents
        
        # evidences 转句子（不添加任何IDK句子）
        evid_sentences = convert_evidences_to_sentences(question_sample.get('evidences', '[]'), converter)
        
        unanswerable_sample = {
            'question': question,
            'documents': str(modified_documents),
            'answer': answer,
            'data_source': question_sample.get('data_source', '2wikimultihop'),
            'evidences': evid_sentences,
            'extra_info': question_sample.get('extra_info', {}),
        }
        
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
            kept_samples.append(unanswerable_sample)
            pbar.update(1)
            pbar.set_postfix({
                '保留': len(kept_samples),
                '移除': removed_count,
                '处理': processed_count,
                '移除率': f'{(removed_count/processed_count*100):.1f}%' if processed_count > 0 else '0.0%'
            })
        else:
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
    print(f"移除率: {(removed_count/processed_count*100):.1f}%" if processed_count > 0 else "移除率: 0.0%")
    
    return kept_samples


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description='2wikimultihop 转化 + evidences句子化（仅保存answerable=true样本）')
    parser.add_argument('--type', type=str, default='train', help='train或test')
    parser.add_argument('--template_type', type=str, default='deepseek-r1-distill-qwen')
    parser.add_argument('--size', type=int, required=True, help='目标样本总数（所有样本均为answerable=true）')
    
    # 数据路径
    parser.add_argument('--data-path', type=str, default=None, help='输入JSONL文件路径')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("2WIKIMULTIHOP 转化 + EVIDENCES句子化")
    print("="*80)
    print(f"总目标样本数: {args.size}")
    print(f"  - Answerable(True): {args.size}")
    print("="*80)
    
    # 确定数据路径
    if args.data_path:
        data_path = args.data_path
    elif args.type == 'train':
        data_path = '/mnt/shared-storage-user/liyafu/runquan/2wikimultihop/data/train.jsonl'
    else:
        data_path = '/mnt/shared-storage-user/liyafu/runquan/2wikimultihop/data/dev.jsonl'
    
    # 加载原始数据集
    print(f"\n📂 从{data_path}加载数据...")
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': data_path})
    print(f"   ✓ 原始数据集长度: {len(raw_dataset)}")
    
    # 打乱并选择足够的样本
    # 为unanswerable预留更多样本以提升过滤成功率
    total_needed = answerable_size + unanswerable_size * 10
    dataset = raw_dataset.shuffle(seed=42).select(range(min(total_needed, len(raw_dataset))))
    print(f"   ✓ 选择了{len(dataset)}个样本用于处理")
    
    # 无需初始化Best-of-N过滤相关的模型和后处理器
    
    # 初始化三元组->句子转换器（使用环境变量配置API）
    converter = TripleSentenceConverter()
    if converter.use_api:
        print("   ✓ Evidence转换将使用Chat Completions API")
    else:
        print("   ⚠️ Evidence转换将使用回退模板（未配置API或requests不可用）")
    
    # 创建 answerable=True 样本并保存
    answerable_samples = create_answerable_samples(dataset, args.size, args.template_type, converter)
    
    # 生成prompt并保存为文件
    def build_row_with_prompt(example: dict) -> dict:
        question_prefixed = make_prefix_unified(example, template_type=args.template_type)
        return {
            "prompt": question_prefixed,
            "question": example['question'],
            "answer": example['answer'],
            "data_source": example['data_source'],
            "extra_info": example['extra_info'],
            "documents": example['documents'],
            "evidences": example['evidences'],
        }
    
    answerable_ds = Dataset.from_list(answerable_samples)
    print("\n为answerable=True样本生成prompt...")
    answerable_ds = answerable_ds.map(lambda ex, idx: build_row_with_prompt(ex), with_indices=True)
    
    output_dir = f'data/2wikimultihop/{args.template_type}'
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    
    # 根据类型保存为train或test文件
    if args.type == 'train':
        output_path = os.path.join(output_dir, 'train.parquet')
    else:
        output_path = os.path.join(output_dir, 'test.parquet')
    
    answerable_ds.to_parquet(output_path)
    print(f"💾 已保存到 {output_path} (样本数: {len(answerable_ds)})")
    
    # 验证
    print("\n验证保存结果...")
    df = pd.read_parquet(output_path)
    n_true = sum(
        1 for _, row in df.iterrows()
        if isinstance(row.get('extra_info'), (dict, str)) and
        (json.loads(row['extra_info']) if isinstance(row['extra_info'], str) else row['extra_info']).get('answerable') is True
    )
    print(f"   ✓ answerable=True: {n_true}/{len(df)}")
    
    print("\n✅ 完成!\n")


if __name__ == '__main__':
    main()