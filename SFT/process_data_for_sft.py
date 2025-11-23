import pandas as pd
import os

# 输入和输出文件路径
input_file = 'SFT/data/musique_filter.parquet'
output_file = 'SFT/data/musique_for_sft.parquet'

def process_musique_data():
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 读取parquet文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_parquet(input_file)
    
    # 创建新的数据框用于存储处理后的数据
    processed_data = []
    
    # 处理每个样本
    total_samples = len(df)
    print(f"开始处理 {total_samples} 个样本...")
    
    for idx, row in df.iterrows():
        # 提取需要的字段
        question = row.get('question', '')
        answer = row.get('answer', '')
        documents = row.get('documents', '')
        
        # 构建prompt（按照hotpot_with_filter.py中的格式）
        # 将documents格式化为documents_context
        documents_context = documents  # 假设documents已经是格式化好的文本
        
        prompt = f"""**References:**
{documents_context}

**Question:**
{question}"""
        
        # 添加到处理后的数据列表
        processed_data.append({
            'prompt': prompt,
            'response': answer
        })
        
        # 打印进度
        if (idx + 1) % 1000 == 0:
            print(f"已处理 {idx + 1}/{total_samples} 个样本")
    
    # 创建新的DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存为parquet格式
    print(f"正在保存处理后的数据到: {output_file}")
    processed_df.to_parquet(output_file, index=False)
    
    print("数据处理完成！")
    print(f"处理后的数据样本数: {len(processed_df)}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    process_musique_data()