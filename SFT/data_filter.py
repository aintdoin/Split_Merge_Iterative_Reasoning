import os
import sys
import json
import re
import ast
import random
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT_Directly = """You are a helpful assistant. You are given a Question and References. The references may or may not help answer the question. Your task is to answer the question based on factual information in the references or your own knowledge.

Remember: ONLY output the direct answer to the question. DO NOT provide any explanations, reasoning, or additional information. Just give the concise answer."""

def get_system_prompt():
    # 通过环境变量控制使用哪套 system prompt
    # 默认使用 IDK_AWARE，可通过环境变量 SYSTEM_PROMPT_TYPE 切换
    return SYSTEM_PROMPT_Directly

def wrap_prompt_with_system(user_content: str, model_template: str = 'qwen') -> str:
    """
    将用户内容包装为完整的 chat 格式（包含 system prompt）
    
    Args:
        user_content: 用户提示内容（question + documents 等）
        model_template: 模型模板类型 ('qwen', 'llama', 等)
    
    Returns:
        完整的 prompt 字符串
    """
    system_prompt = get_system_prompt()
    
    if model_template in ['llama']:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""


# 可选导入 - requests for API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("API unavaiable!")


def make_prefix(dp, template_type='qwen'):
    if 'question' not in dp:
        raise ValueError("question unavaiable!")
    question = dp['question']
    
    if 'documents' not in dp:
        raise ValueError("documents unavaiable!")
    documents_str = dp['documents']
    
    # 解析并格式化documents
    try:
        documents_list = ast.literal_eval(documents_str)
        if not documents_list:
            raise ValueError("documents unavaiable!")
    except Exception as e:
        if isinstance(e, ValueError) and str(e) == "documents为空列表":
            raise
        raise ValueError(f"解析documents失败: {str(e)}")
    
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
    
    if not formatted_docs:
        raise ValueError("无法从documents中提取有效的参考信息")
    documents_context = "\n".join(formatted_docs)
    
    # 用户内容块
    user_content = f"""**References:**
{documents_context}

**Question:**
{question}"""
    
    prompt = wrap_prompt_with_system(user_content, template_type)
    
    return prompt


def _has_valid_format(text: str) -> bool:
    """检查文本是否有有效的格式"""
    try:
        # 这里可以根据实际需求修改格式检查逻辑
        return bool(text and text.strip())
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
    
    try:
        resp = requests.post(chat_url, json=chat_payload, headers=headers, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            candidates = []
            for choice in data.get('choices', []):
                text = choice.get('message', {}).get('content', '')
                if text and text.strip():
                    candidates.append(text.strip())
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
    except Exception as e:
        print(f"\n⚠️  API调用异常: {str(e)}")
        return []


# 从verl.utils导入AnswerPostProcessor
from verl.utils.reward_score.answer_postprocessor import get_postprocessor


def evaluate_sample_accuracy(sample_dict, prompt, args, llm, sampling_params, postprocessor):
    """
    评估单个样本的回答正确率
    返回: (correctness_rate: float, correct_count: int, total_count: int)
    """
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
    
    # 所有可能的答案
    all_answers = [ground_truth]
    if answer_aliases and len(answer_aliases) > 0:
        all_answers.extend(answer_aliases)
    
    # 生成N个候选回答（统一使用 vLLM）
    candidates = []
    try:
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        output = outputs[0]
        candidates = [out.text for out in output.outputs]
    except Exception:
        # vLLM 调用失败，静默返回
        return 0.0, 0, 0
    
    # 评估所有候选
    correct_count = 0
    total_count = len(candidates)
    
    for candidate_text in candidates:
        if not _has_valid_format(candidate_text):
            continue
        
        # 检查是否是IDK回答（得0分）
        if _is_idk_answer(candidate_text):
            continue
        
        # 尝试所有可能的答案
        is_correct = False
        for ans in all_answers:
            if ans:
                # 使用原有的本地 AnswerPostProcessor 判定
                formatted_answer = f"</think><think></think>{candidate_text}</answer>"
                score = postprocessor.judge_answer_correctness(
                    predicted_answer=formatted_answer,
                    ground_truth_answer=ans,
                    question=question,
                    answerable=True
                )
                if score >= 0.999:
                    is_correct = True
                    break
        if is_correct:
            correct_count += 1
    
    # 计算正确率
    correctness_rate = correct_count / total_count if total_count > 0 else 0.0
    
    return correctness_rate, correct_count, total_count


def load_musique_dataset(data_path, sample_size=None):
    """加载musique数据集"""
    print(f"加载数据集: {data_path}")
    
    # 读取JSONL文件
    all_records = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 确保必要字段存在
                if 'question' not in data:
                    continue
                if 'answer' not in data:
                    data['answer'] = ''
                if 'paragraphs' in data:
                    paragraphs = data['paragraphs'] if isinstance(data['paragraphs'], list) else []
                    documents = []
                    for para in paragraphs:
                        title = para.get('title', '')
                        paragraph_text = para.get('paragraph_text', '')
                        documents.append([title, [paragraph_text]])
                    data['documents'] = str(documents)
                    del data['paragraphs']
                if 'data_source' not in data:
                    data['data_source'] = 'musique'
                if 'evidences' not in data:
                    data['evidences'] = '[]'
                if 'extra_info' not in data:
                    data['extra_info'] = {}
                if '_id' in data:
                    if isinstance(data['extra_info'], dict):
                        data['extra_info']['sample_id'] = data['_id']
                
                all_records.append(data)
            except Exception as e:
                print(f"  解析行时出错: {str(e)}")
                continue
    
    print(f"总共加载 {len(all_records)} 条记录")
    
    # 抽样
    if sample_size is not None and sample_size < len(all_records):
        random.seed(42)
        all_records = random.sample(all_records, sample_size)
        print(f"抽样后保留 {len(all_records)} 条记录")
    
    return Dataset.from_list(all_records)


def main():
    parser = argparse.ArgumentParser(description='musique数据集过滤')
    parser.add_argument('--data-path', type=str, required=True, help='输入数据文件路径')
    parser.add_argument('--output-path', type=str, default='musique_filter.parquet', help='输出文件路径')
    parser.add_argument('--sample-size', type=int, default=None, help='抽样大小')
    
    # 模型/API配置
    parser.add_argument('--model-path', type=str, default='', help='本地模型路径（vLLM模式）')
    parser.add_argument('--use-api', action='store_true', help='使用API模式而非本地vLLM')
    parser.add_argument('--api-base', type=str, default='http://localhost:8000', help='API基础URL')
    parser.add_argument('--api-key', type=str, default='', help='API密钥（可选）')
    parser.add_argument('--model-name', type=str, default='', help='API模型名称')
    parser.add_argument('--template-type', type=str, default='qwen', help='模型模板类型 (qwen, llama等)')
    
    # 过滤参数
    parser.add_argument('--n-candidates', type=int, default=32, help='每个样本生成的候选回答数量')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--do-sample', type=bool, default=True, help='是否进行采样')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p采样参数')
    parser.add_argument('--top-k', type=int, default=100, help='Top-k采样参数')
    parser.add_argument('--max-tokens', type=int, default=2048, help='最大生成token数')
    
    # 阈值参数
    parser.add_argument('--upper', type=float, default=0.9, help='高正确率阈值')
    parser.add_argument('--lower', type=float, default=0.1, help='低正确率阈值')
    
    # LLM Judge配置
    parser.add_argument('--judge-api-base', type=str, default=None, help='Judge API基础URL')
    parser.add_argument('--judge-api-key', type=str, default=None, help='Judge API密钥')
    parser.add_argument('--judge-model-name', type=str, default='llm-judge', help='Judge模型名称')
    
    args = parser.parse_args()
    
    # 初始化LLM和sampling_params
    llm = None
    sampling_params = None
    
    # 始终通过 vLLM 进行候选生成
    try:
        from vllm import LLM, SamplingParams
        print(f"加载本地模型: {args.model_path}")
        llm = LLM(model=args.model_path, tensor_parallel_size=1)
        sampling_params = SamplingParams(
            n=args.n_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens
        )
    except ImportError:
        print("错误: vllm库不可用，请安装vllm以进行候选生成")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载模型失败: {str(e)}")
        sys.exit(1)
    
    # 初始化Answer PostProcessor
    print("初始化answer postprocessor...")
    postprocessor = get_postprocessor()
    print("✓ Postprocessor初始化完成")
    
    # 加载数据集
    dataset = load_musique_dataset(args.data_path, args.sample_size)
    
    # 处理样本
    high_confidence_samples = []  # 正确率 > upper
    low_confidence_samples = []   # 正确率 < lower
    
    stats = {
        'total_processed': 0,
        'high_confidence': 0,
        'low_confidence': 0,
        'medium_confidence': 0,
        'skipped': 0
    }
    
    # 初始化进度条
    print(f"\n{'='*80}")
    print(f"开始处理样本，使用Best-of-{args.n_candidates}策略")
    print(f"高正确率阈值: > {args.upper}")
    print(f"低正确率阈值: < {args.lower}")
    print(f"{'='*80}")
    
    # 使用tqdm进度条处理样本
    with tqdm(total=len(dataset), desc="处理进度", unit="样本") as pbar:
        for i in range(len(dataset)):
            sample = dataset[i]
            try:
                # 生成prompt
                prompt = make_prefix(sample, args.template_type)
                
                # 评估正确率（vLLM生成 + Judge API判定）
                correctness_rate, correct_count, total_count = evaluate_sample_accuracy(
                    sample, prompt, args, llm, sampling_params, postprocessor
                )
                
                # 处理样本分类
                if total_count == 0:
                    stats['skipped'] += 1
                else:
                    processed_sample = sample.copy()
                    processed_sample['correctness_rate'] = correctness_rate
                    processed_sample['correct_count'] = correct_count
                    processed_sample['total_count'] = total_count
                    
                    if correctness_rate > args.upper:
                        high_confidence_samples.append(processed_sample)
                        stats['high_confidence'] += 1
                    elif correctness_rate < args.lower:
                        processed_sample['answer'] = "I don't know"
                        if isinstance(processed_sample.get('extra_info'), dict):
                            processed_sample['extra_info']['original_answer'] = sample.get('answer', '')
                        low_confidence_samples.append(processed_sample)
                        stats['low_confidence'] += 1
                    else:
                        stats['medium_confidence'] += 1
                    
                    stats['total_processed'] += 1
                
                # 更新进度条信息
                pbar.set_postfix({
                    '高': stats['high_confidence'],
                    '中': stats['medium_confidence'],
                    '低': stats['low_confidence'],
                    '跳过': stats['skipped']
                })
                pbar.update(1)
            except Exception:
                # 不打印错误信息，但更新统计
                stats['skipped'] += 1
                # 更新进度条
                pbar.set_postfix({
                    '高': stats['high_confidence'],
                    '中': stats['medium_confidence'],
                    '低': stats['low_confidence'],
                    '跳过': stats['skipped']
                })
                pbar.update(1)

    
    # 合并两类样本
    filtered_samples = high_confidence_samples + low_confidence_samples
    print(f"\n{'='*80}")
    print("处理结果统计:")
    print(f"总处理样本数: {stats['total_processed']}")
    print(f"高正确率样本数 (> {args.upper}): {stats['high_confidence']}")
    print(f"中等正确率样本数 ({args.lower}-{args.upper}): {stats['medium_confidence']}")
    print(f"低正确率样本数 (< {args.lower}): {stats['low_confidence']}")
    print(f"跳过样本数: {stats['skipped']}")
    print(f"最终保留样本数: {len(filtered_samples)}")
    print(f"{'='*80}")
    
    # 保存结果
    if filtered_samples:
        filtered_dataset = Dataset.from_list(filtered_samples)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        filtered_dataset.to_parquet(args.output_path)
        print(f"\n✓ 保存结果到: {args.output_path}")
        print(f"  样本数: {len(filtered_dataset)}")
    else:
        print("\n⚠️  没有样本通过过滤，未保存结果")


if __name__ == '__main__':
    main()