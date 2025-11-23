def extract_jsonl_by_line_number(jsonl_path, target_line):
    """
    按行号提取 JSONL 文件中的某一行
    :param jsonl_path: JSONL 文件路径（绝对路径或相对路径）
    :param target_line: 目标行号（从 1 开始计数）
    :return: 目标行的原始字符串（或解析后的 JSON 对象）
    """
    try:
        # 读取所有行（小文件推荐，简洁高效）
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 校验行号合法性
        if target_line < 1 or target_line > len(lines):
            return f"错误：文件仅包含 {len(lines)} 行，目标行号 {target_line} 超出范围"
        
        # 提取目标行（去除换行符）
        target_content = lines[target_line - 1].strip()
        
        # 可选：解析为 JSON 对象（如果需要结构化数据）
        import json
        try:
            json_obj = json.loads(target_content)
            return json_obj  # 返回解析后的字典
        except json.JSONDecodeError:
            return f"警告：第 {target_line} 行不是合法 JSON，原始内容：{target_content}"
    
    except FileNotFoundError:
        return f"错误：文件 {jsonl_path} 不存在"
    except Exception as e:
        return f"错误：{str(e)}"


# ------------------- 用法示例 -------------------
if __name__ == "__main__":
    # 1. 配置文件路径和目标行号
    jsonl_file = '/mnt/shared-storage-user/liyafu/runquan/hotpot/hotpot_train_v1.1.jsonl'  # 你的 JSONL 文件路径
    line_number = 3  # 要提取的行号（从 1 开始）
    
    # 2. 调用函数提取
    result = extract_jsonl_by_line_number(jsonl_file, line_number)
    
    # 3. 打印结果
    print("提取结果：")
    print(result)