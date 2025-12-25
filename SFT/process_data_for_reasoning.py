# /Users/guirunquan/Documents/github/Split_Merge_Iterative_Reasoning/SFT/process_data_for_reasoning.py
import pandas as pd
import numpy as np
import os

# 输入和输出文件路径
input_file = 'data/2wikimultihop/train.parquet'  # 可以根据需要修改
output_file = 'SFT/data/2wikimultihop_train.parquet'  # SFT数据
output_rl_file = 'data/2wikimultihop/train_rl.parquet'  # RL数据(覆盖原文件或另存)

def process_data_for_reasoning():
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 读取parquet文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_parquet(input_file)
    
    # 打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 计算切分点
    split_idx = int(len(df) * 0.7)
    
    # 创建列表用于存储处理后的数据
    sft_data = []
    rl_data = []
    
    # 处理每个样本
    total_samples = len(df)
    print(f"开始处理 {total_samples} 个样本 (70% SFT, 30% RL)...")
    
    for idx, row in df.iterrows():
        # 提取需要的字段
        prompt = row.get('prompt', '')
        answer = row.get('answer', '')
        evidences = row.get('evidences', [])
        
        # 构建reasoning部分
        reasoning_lines = []
        # 检查evidences是否为字符串格式的列表，如 "['item1', 'item2']"
        if isinstance(evidences, str) and evidences.startswith('[') and evidences.endswith(']'):
            try:
                # 尝试将字符串解析为列表
                import ast
                parsed_evidences = ast.literal_eval(evidences)
                if isinstance(parsed_evidences, list):
                    for i, evidence in enumerate(parsed_evidences):
                        reasoning_lines.append(f"{i+1}. {evidence}")
            except (ValueError, SyntaxError):
                # 如果解析失败，将整个字符串作为一个推理步骤
                reasoning_lines.append(f"1. {evidences}")
        elif isinstance(evidences, (list, np.ndarray)):
            # 正常的列表处理
            for i, evidence in enumerate(evidences):
                reasoning_lines.append(f"{i+1}. {evidence}")
        else:
            # 其他情况，将其作为单个推理步骤
            reasoning_lines.append(f"1. {evidences}")
        reasoning_text = "\n".join(reasoning_lines)
        hops = len(reasoning_lines)
        
        if idx < split_idx:
            # SFT数据处理 (70%)
            # 构建response
            response = f"""<think>
{reasoning_text}
</think>
<answer>{answer}</answer>"""
            
            # 添加到SFT数据列表
            sft_data.append({
                'prompt': prompt,
                'response': response,
                'hops': hops
            })
        else:
            # RL数据处理 (30%)
            # 保留原始格式，增加hops
            row_dict = row.to_dict()
            row_dict['hops'] = hops
            rl_data.append(row_dict)
        
        # 打印进度
        if (idx + 1) % 1000 == 0:
            print(f"已处理 {idx + 1}/{total_samples} 个样本")
    
    # 保存SFT数据
    sft_df = pd.DataFrame(sft_data)
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"正在保存SFT数据到: {output_file}")
    sft_df.to_parquet(output_file, index=False)
    
    # 保存RL数据
    rl_df = pd.DataFrame(rl_data)
    # 确保输出目录存在
    output_rl_dir = os.path.dirname(output_rl_file)
    if output_rl_dir and not os.path.exists(output_rl_dir):
        os.makedirs(output_rl_dir)
    print(f"正在保存RL数据到: {output_rl_file}")
    rl_df.to_parquet(output_rl_file, index=False)
    
    print("数据处理完成！")
    print(f"SFT数据样本数: {len(sft_df)}")
    print(f"RL数据样本数: {len(rl_df)}")

if __name__ == "__main__":
    process_data_for_reasoning()