#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取古汉语常用词样例.md中的文本块
将每个```text和```之间的内容保存为单独的txt文件
"""

import re
import os
from pathlib import Path

def extract_text_blocks(markdown_file: str, output_dir: str = "extracted_texts"):
    """
    从Markdown文件中提取```text和```之间的内容
    
    Args:
        markdown_file: Markdown文件路径
        output_dir: 输出目录
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"开始提取文件: {markdown_file}")
    print(f"输出目录: {output_path.absolute()}")
    
    # 读取Markdown文件
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配```text和```之间的内容
    # 模式：```text(.*?)```
    pattern = r'```text\s*(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    print(f"找到 {len(matches)} 个文本块")
    
    # 保存每个文本块
    for i, text_block in enumerate(matches, 1):
        # 清理文本块
        text_block = text_block.strip()
        
        # 从文本块中提取词条编号和词名
        first_line = text_block.split('\n')[0] if text_block else ""
        word_match = re.search(r'(\d+)\.【(.+)】', first_line)
        
        if word_match:
            word_number = word_match.group(1)
            word_name = word_match.group(2)
            filename = f"{word_number.zfill(2)}_{word_name}.txt"
        else:
            # 如果没有找到词条编号，使用序号
            filename = f"word_{i:02d}.txt"
        
        # 完整的文件路径
        file_path = output_path / filename
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_block)
        
        print(f"保存: {filename}")
    
    print(f"\n提取完成！共保存 {len(matches)} 个文件到 {output_path.absolute()}")

def main():
    """主函数"""
    # 输入文件
    input_file = "古汉语常用词样例.md"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        return
    
    # 输出目录
    output_dir = "extracted_texts"
    
    # 提取文本块
    extract_text_blocks(input_file, output_dir)
    
    # 显示提取的文件列表
    print("\n提取的文件列表:")
    output_path = Path(output_dir)
    for file in sorted(output_path.glob("*.txt")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()