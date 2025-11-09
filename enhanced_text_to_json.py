#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型增强的古汉语文本到JSON转换器
基于规则解析，并使用大模型API优化解析结果
支持DeepSeek等大模型API
"""

import re
import json
import asyncio
import argparse
import os
import glob
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from openai import OpenAI
import requests


@dataclass
class ExtendedMeaning:
    """引申义定义"""
    meaning: str
    examples: List[str] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


@dataclass
class Meaning:
    """词义定义"""
    index: str
    part_of_speech: str
    definition: str
    examples: List[str]
    extended_meanings: List[ExtendedMeaning] = None

    def __post_init__(self):
        if self.extended_meanings is None:
            self.extended_meanings = []


@dataclass
class WordEntry:
    """词条定义"""
    word: str
    pinyin: Optional[str] = None
    meanings: List[Meaning] = None
    comparisons: List[str] = None

    def __post_init__(self):
        if self.meanings is None:
            self.meanings = []
        if self.comparisons is None:
            self.comparisons = []


class LLMEnhancer:
    """大模型增强器，用于优化解析结果"""

    def __init__(self, config: Dict):
        self.config = config
        self.setup_llm_client()

    def setup_llm_client(self):
        """根据配置设置大模型客户端"""
        provider = self.config.get('llm_provider', 'deepseek')

        if provider == 'deepseek':
            self.client = OpenAI(
                api_key=self.config.get('deepseek_api_key'),
                base_url="https://api.deepseek.com"
            )
            self.client_type = 'deepseek'
        elif provider == 'openai':
            self.client = OpenAI(api_key=self.config.get('openai_api_key'))
            self.client_type = 'openai'
        elif provider == 'custom':
            self.client = OpenAI(
                api_key=self.config.get('api_key'),
                base_url=self.config.get('base_url')
            )
            self.client_type = 'custom'
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")

    async def enhance_parsing(self, original_text: str, preliminary_result: Dict) -> Dict:
        """
        使用大模型优化解析结果
        """
        print(f"开始大模型优化词条: {preliminary_result.get('word', '未知')}")
        
        prompt = self.build_enhancement_prompt(original_text, preliminary_result)

        try:
            if self.client_type in ['deepseek', 'openai', 'custom']:
                response = await self.call_openai_compatible(prompt)
            else:
                raise ValueError(f"不支持的客户端类型: {self.client_type}")

            enhanced_result = self.parse_llm_response(response, preliminary_result)
            
            if enhanced_result != preliminary_result:
                print(f"大模型优化成功: {preliminary_result.get('word', '未知')}")
            else:
                print(f"大模型优化使用原始结果: {preliminary_result.get('word', '未知')}")
                
            return enhanced_result

        except Exception as e:
            print(f"大模型优化失败: {e}")
            print(f"词条: {preliminary_result.get('word', '未知')}")
            # 失败时返回原始结果
            return preliminary_result

    def build_enhancement_prompt(self, original_text: str, preliminary_result: Dict) -> str:
        """构建优化提示词"""
        prompt = f"""
你是一个古汉语专家，请帮助校验和优化古汉语词条的解析结果。

原始文本：
{original_text}

初步解析结果（可能存在错误）：
{json.dumps(preliminary_result, ensure_ascii=False, indent=2)}

请完成以下任务：
1. 检查解析结果是否正确识别了所有义项
2. 验证词性标注是否准确
3. 确认例句是否完整提取
4. 检查引申义是否正确识别
5. 如果有词义辨析内容，请提取到comparisons字段
6. 如果可能，补充拼音信息

重要要求：
- 只输出修正后的完整JSON格式，保持与输入相同的结构
- 不要输出任何其他文字、解释或注释
- 确保JSON格式完全正确，可以直接被json.loads()解析
- 保持字段名称和结构不变

请严格按照以下JSON格式输出：
{{
  "word": "词",
  "pinyin": "拼音（可选）",
  "meanings": [
    {{
      "index": "一",
      "part_of_speech": "词性",
      "definition": "定义",
      "examples": ["例句1", "例句2"],
      "extended_meanings": [
        {{
          "meaning": "引申义",
          "examples": ["例句1", "例句2"]
        }}
      ]
    }}
  ],
  "comparisons": ["辨析内容"]
}}
"""
        return prompt

    async def call_openai_compatible(self, prompt: str) -> str:
        """调用OpenAI兼容的API"""
        model_map = {
            'deepseek': 'deepseek-chat',
            'openai': 'gpt-4',
            'custom': self.config.get('model', 'deepseek-chat')
        }
        
        model = model_map.get(self.client_type, 'deepseek-chat')
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个古汉语专家，擅长解析古汉语文本结构。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 低温度确保输出稳定
            max_tokens=2000,
            stream=False
        )
        return response.choices[0].message.content

    def parse_llm_response(self, response: str, fallback_result: Dict) -> Dict:
        """解析大模型响应"""
        try:
            # 清理响应，移除可能的重复内容
            response = response.strip()
            
            # 尝试直接解析整个响应
            try:
                result = json.loads(response)
                if self.validate_enhanced_result(result):
                    return result
            except json.JSONDecodeError:
                pass
            
            # 如果直接解析失败，尝试提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                if self.validate_enhanced_result(result):
                    return result
            
            print("大模型响应中未找到有效的JSON")
            return fallback_result
            
        except json.JSONDecodeError as e:
            print(f"大模型响应JSON解析失败: {e}")
            print(f"响应内容: {response[:500]}...")  # 只打印前500字符用于调试
            return fallback_result
        except Exception as e:
            print(f"大模型响应解析异常: {e}")
            return fallback_result
    
    def validate_enhanced_result(self, result: Dict) -> bool:
        """验证大模型增强结果的格式"""
        try:
            # 检查基本结构
            if not isinstance(result, dict):
                return False
            if 'word' not in result:
                return False
            if 'meanings' not in result or not isinstance(result['meanings'], list):
                return False
            
            # 检查每个义项的结构
            for meaning in result['meanings']:
                if not isinstance(meaning, dict):
                    return False
                if 'index' not in meaning or 'part_of_speech' not in meaning:
                    return False
                if 'definition' not in meaning or 'examples' not in meaning:
                    return False
                if not isinstance(meaning['examples'], list):
                    return False
                
                # 检查引申义结构
                if 'extended_meanings' in meaning:
                    if not isinstance(meaning['extended_meanings'], list):
                        return False
                    for ext_meaning in meaning['extended_meanings']:
                        if not isinstance(ext_meaning, dict):
                            return False
                        if 'meaning' not in ext_meaning or 'examples' not in ext_meaning:
                            return False
            
            return True
        except Exception:
            return False


class ClassicalChineseTextToJsonConverter:
    """将古汉语文本转换为JSON的转换器"""

    def __init__(self):
        # 匹配词条：1.【言】
        self.word_pattern = re.compile(r'(\d+)\.【(.+)】')
        # 匹配义项：（一）动词。说话，说。
        self.meaning_pattern = re.compile(r'（([一二三四五六七八九十]+)）\s*([^。]+)。\s*(.+)')
        # 匹配例句：《书名》："例句"
        self.example_pattern = re.compile(r'《([^》]+)》："([^"]+)"')
        # 匹配引申义：引申爲[...]
        self.extended_pattern = re.compile(r'引申爲\[([^\]]+)\]')
        # 匹配辨析：［辨］...
        self.comparison_pattern = re.compile(r'［辨］(.+)')
        # 匹配读音：讀yù
        self.pronunciation_pattern = re.compile(r'讀([a-zà-ÿ]+)')
        # 匹配注释：（注释内容）
        self.note_pattern = re.compile(r'（([^）]+)）')

    def convert_text_to_json(self, text: str) -> List[Dict]:
        """将完整文本转换为JSON格式的词条列表"""
        # 分割词条
        entries = self.split_into_entries(text)

        json_entries = []
        for entry_text in entries:
            word_entry = self.parse_single_entry(entry_text)
            if word_entry:
                json_entries.append(asdict(word_entry))

        return json_entries

    def split_into_entries(self, text: str) -> List[str]:
        """将文本分割为单个词条的文本块"""
        entries = []
        current_entry = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否是新的词条开始
            if self.word_pattern.match(line):
                # 如果已经有当前词条，保存它
                if current_entry:
                    entries.append('\n'.join(current_entry))
                    current_entry = []

            current_entry.append(line)

        # 添加最后一个词条
        if current_entry:
            entries.append('\n'.join(current_entry))

        return entries

    def parse_single_entry(self, entry_text: str) -> Optional[WordEntry]:
        """解析单个词条文本"""
        lines = entry_text.strip().split('\n')
        if not lines:
            return None

        # 解析词条标题
        title_line = lines[0]
        word_match = self.word_pattern.match(title_line)
        if not word_match:
            return None

        word_number = word_match.group(1)
        word = word_match.group(2)

        # 初始化词条对象
        word_entry = WordEntry(word=word, meanings=[], comparisons=[])

        current_meaning = None
        current_definition = ""

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # 检查是否是新的义项
            meaning_match = self.meaning_pattern.match(line)
            if meaning_match:
                # 保存上一个义项
                if current_meaning:
                    # 处理当前定义中的例句和引申义
                    self.process_definition(current_meaning, current_definition)
                    word_entry.meanings.append(current_meaning)

                # 开始新义项
                index = meaning_match.group(1)
                pos = meaning_match.group(2)
                definition_start = meaning_match.group(3)

                current_meaning = Meaning(
                    index=index,
                    part_of_speech=pos,
                    definition="",
                    examples=[],
                    extended_meanings=[]
                )
                current_definition = definition_start
            elif current_meaning:
                # 继续累积定义文本
                current_definition += " " + line
            elif "［辨］" in line:
                # 处理词义辨析
                comparison_match = self.comparison_pattern.search(line)
                if comparison_match:
                    word_entry.comparisons.append(comparison_match.group(1))

        # 处理最后一个义项
        if current_meaning:
            self.process_definition(current_meaning, current_definition)
            word_entry.meanings.append(current_meaning)

        return word_entry

    def process_definition(self, meaning: Meaning, definition_text: str):
        """处理定义文本，提取例句和引申义"""
        # 先提取基本定义部分（在第一个引申义之前）
        basic_definition = self.extract_basic_definition(definition_text)
        meaning.definition = basic_definition
        
        # 提取基本定义对应的例句（从完整文本中提取，但在第一个引申义之前）
        basic_examples = self.extract_examples_before_extended(definition_text)
        meaning.examples.extend(basic_examples)

        # 提取引申义及其对应例句（从完整文本中提取，但排除基本定义中的例句）
        extended_meanings = self.extract_extended_meanings_with_examples(definition_text, basic_examples)
        meaning.extended_meanings.extend(extended_meanings)

    def extract_examples(self, text: str) -> List[str]:
        """从文本中提取例句"""
        examples = []
        # 使用更灵活的模式来匹配包含书名的完整段落
        # 匹配从《书名》开始，到下一个《书名》或文本结束的内容
        pattern = r'《[^》]+》[^《]*'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # 检查是否包含书名和引号内的内容
            if '《' in match and '》' in match and '“' in match:
                # 清理并添加
                example_text = match.strip()
                examples.append(example_text)
        
        return examples
    
    def extract_examples_before_extended(self, text: str) -> List[str]:
        """提取第一个引申义之前的例句"""
        # 找到第一个引申义标记的位置
        first_extended = re.search(r'引申爲\[[^\]]+\]', text)
        first_you = re.search(r'又\[[^\]]+\]', text)
        
        # 找到最早出现的引申义位置
        positions = []
        if first_extended:
            positions.append(first_extended.start())
        if first_you:
            positions.append(first_you.start())
        
        if positions:
            # 截取到第一个引申义之前的内容
            cut_position = min(positions)
            text_before_extended = text[:cut_position]
        else:
            # 没有引申义，返回完整文本
            text_before_extended = text
        
        # 从截取后的文本中提取例句
        return self.extract_examples(text_before_extended)

    def extract_basic_definition(self, text: str) -> str:
        """提取基本定义部分（在第一个引申义之前）"""
        # 找到第一个引申义标记的位置
        first_extended = re.search(r'引申爲\[[^\]]+\]', text)
        first_you = re.search(r'又\[[^\]]+\]', text)
        
        # 找到最早出现的引申义位置
        positions = []
        if first_extended:
            positions.append(first_extended.start())
        if first_you:
            positions.append(first_you.start())
        
        if positions:
            # 截取到第一个引申义之前的内容
            cut_position = min(positions)
            basic_text = text[:cut_position]
        else:
            # 没有引申义，返回完整文本
            basic_text = text
        
        # 提取基本定义 - 只保留定义本身，移除例句
        # 找到第一个例句的位置
        first_example = re.search(r'《[^》]+》：“[^《]+”', basic_text)
        if first_example:
            # 截取到第一个例句之前的内容
            basic_text = basic_text[:first_example.start()]
        
        # 清理文本 - 只移除注释，保留其他内容
        basic_text = self.note_pattern.sub('', basic_text)
        basic_text = re.sub(r'\s+', ' ', basic_text).strip()
        basic_text = re.sub(r'[。，；、]\s*$', '', basic_text)
        return basic_text
    
    def extract_extended_meanings_with_examples(self, text: str, basic_examples: List[str]) -> List[ExtendedMeaning]:
        """从文本中提取引申义及其对应例句，排除基本定义中的例句"""
        extended_meanings = []
        
        # 使用更精确的方法：找到所有引申义标记的位置
        extended_positions = []
        
        # 找到所有"引申爲[...]"的位置
        for match in re.finditer(r'引申爲\[([^\]]+)\]', text):
            extended_positions.append(('引申爲', match.group(1), match.start(), match.end()))
        
        # 找到所有"又[...]"的位置
        for match in re.finditer(r'又\[([^\]]+)\]', text):
            extended_positions.append(('又', match.group(1), match.start(), match.end()))
        
        # 按位置排序
        extended_positions.sort(key=lambda x: x[2])
        
        # 为每个引申义提取对应的例句
        for i, (ext_type, meaning, start, end) in enumerate(extended_positions):
            # 确定该引申义对应的文本范围
            if i < len(extended_positions) - 1:
                # 到下一个引申义之前
                next_start = extended_positions[i+1][2]
                following_text = text[end:next_start]
            else:
                # 最后一个引申义，到文本结束
                following_text = text[end:]
            # 提取该引申义对应的例句 - 只提取紧接着的例句，直到下一个《》出现
            examples = self.extract_examples_until_next_book(following_text)
            # 过滤掉基本定义中已经存在的例句
            filtered_examples = [ex for ex in examples if ex not in basic_examples]
            extended_meanings.append(ExtendedMeaning(meaning=meaning, examples=filtered_examples))
        
        return extended_meanings
    
    def extract_examples_until_next_book(self, text: str) -> List[str]:
        """提取例句，直到遇到下一个引申义标记或文本结束"""
        examples = []
        
        # 使用新的提取逻辑获取所有例句
        all_examples = self.extract_examples(text)
        
        # 检查是否有引申义标记，如果有则只取第一个引申义之前的例句
        first_extended = re.search(r'引申爲\[[^\]]+\]|又\[[^\]]+\]', text)
        if first_extended:
            # 找到第一个引申义的位置
            extended_pos = first_extended.start()
            # 只保留在第一个引申义之前的例句
            for example in all_examples:
                # 检查这个例句是否在第一个引申义之前
                example_pos = text.find(example)
                if example_pos < extended_pos:
                    examples.append(example)
                else:
                    break
        else:
            # 没有引申义，返回所有例句
            examples = all_examples
        
        return examples

    def clean_definition_text(self, text: str) -> str:
        """清理定义文本，移除例句和特殊标记"""
        # 移除例句
        text = self.example_pattern.sub('', text)
        # 移除引申义标记
        text = self.extended_pattern.sub('', text)
        # 移除"又[...]"标记
        text = re.sub(r'又\[[^\]]+\]', '', text)
        # 移除注释
        text = self.note_pattern.sub('', text)
        # 清理多余空格和标点
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[。，；、]\s*$', '', text)
        return text


class EnhancedClassicalChineseConverter(ClassicalChineseTextToJsonConverter):
    """增强的古汉语文本转换器，集成大模型优化"""

    def __init__(self, llm_config: Dict = None):
        super().__init__()
        self.llm_enhancer = None
        if llm_config and llm_config.get('enable_llm', False):
            self.llm_enhancer = LLMEnhancer(llm_config)

        # 置信度阈值配置
        self.confidence_threshold = llm_config.get('confidence_threshold', 0.7) if llm_config else 0.7

    async def convert_text_to_json_enhanced(self, text: str) -> List[Dict]:
        """增强的文本转换方法，集成大模型优化"""
        # 首先使用规则方法进行初步解析
        preliminary_entries = self.convert_text_to_json(text)

        if not self.llm_enhancer:
            return preliminary_entries

        # 对每个词条进行大模型优化
        enhanced_entries = []
        for entry in preliminary_entries:
            # 计算当前解析的置信度
            confidence = self.calculate_confidence(entry)

            if confidence < self.confidence_threshold:
                # 低置信度条目使用大模型优化
                original_text = self.find_original_text(text, entry['word'])
                enhanced_entry = await self.llm_enhancer.enhance_parsing(
                    original_text, entry
                )
                enhanced_entries.append(enhanced_entry)
            else:
                # 高置信度条目直接使用
                enhanced_entries.append(entry)

        return enhanced_entries

    def calculate_confidence(self, entry: Dict) -> float:
        """计算解析结果的置信度"""
        confidence = 1.0

        # 检查必要字段
        if not entry.get('word'):
            confidence *= 0.5

        if not entry.get('meanings') or len(entry['meanings']) == 0:
            confidence *= 0.3
        else:
            # 检查每个义项的完整性
            for meaning in entry['meanings']:
                if not meaning.get('definition'):
                    confidence *= 0.8
                if not meaning.get('examples') or len(meaning['examples']) == 0:
                    confidence *= 0.9
                if not meaning.get('part_of_speech'):
                    confidence *= 0.7

        return confidence

    def find_original_text(self, full_text: str, word: str) -> str:
        """从完整文本中查找特定词的原始文本"""
        lines = full_text.split('\n')
        entry_lines = []
        in_target_entry = False

        for line in lines:
            if f"【{word}】" in line:
                in_target_entry = True
                entry_lines = [line]  # 重新开始
            elif in_target_entry:
                if line.strip() and not line.startswith((' ', '\t')) and '【' in line:
                    # 新的词条开始
                    break
                entry_lines.append(line)

        return '\n'.join(entry_lines)

    def batch_convert_with_enhancement(self, input_dir: str, output_dir: str):
        """批量转换增强版本"""
        import asyncio
        import os
        import glob

        text_files = glob.glob(os.path.join(input_dir, "*.txt"))

        for text_file in text_files:
            print(f"处理文件: {text_file}")

            with open(text_file, 'r', encoding='utf-8') as f:
                text_content = f.read()

            # 异步处理
            enhanced_data = asyncio.run(self.convert_text_to_json_enhanced(text_content))

            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(text_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_enhanced.json")

            # 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

            # 生成质量报告
            self.generate_quality_report(enhanced_data, output_file)

            print(f"增强转换完成: {text_file} -> {output_file}")

    def generate_quality_report(self, data: List[Dict], output_file: str):
        """生成质量报告"""
        report = {
            'total_entries': len(data),
            'entries_with_issues': 0,
            'issues_details': []
        }

        for entry in data:
            entry_issues = self.validate_entry_quality(entry)
            if entry_issues:
                report['entries_with_issues'] += 1
                report['issues_details'].append({
                    'word': entry['word'],
                    'issues': entry_issues
                })

        # 保存质量报告
        report_file = output_file.replace('.json', '_quality_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def validate_entry_quality(self, entry: Dict) -> List[str]:
        """验证条目质量"""
        issues = []

        if not entry.get('meanings'):
            issues.append("缺少义项")
            return issues

        for i, meaning in enumerate(entry['meanings']):
            if not meaning.get('part_of_speech'):
                issues.append(f"义项 {meaning.get('index', i+1)} 缺少词性")
            if not meaning.get('definition'):
                issues.append(f"义项 {meaning.get('index', i+1)} 缺少定义")
            if not meaning.get('examples'):
                issues.append(f"义项 {meaning.get('index', i+1)} 缺少例句")

        return issues


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='大模型增强的古汉语文本到JSON转换器')

    # 基本参数
    parser.add_argument('input', help='输入文件或目录路径')
    parser.add_argument('-o', '--output', help='输出文件或目录路径')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')

    # 大模型配置参数
    parser.add_argument('--enable-llm', action='store_true', help='启用大模型优化')
    parser.add_argument('--llm-provider', choices=['deepseek', 'openai', 'custom'],
                       default='deepseek', help='大模型提供商')
    parser.add_argument('--llm-model', help='大模型名称')
    parser.add_argument('--api-key', help='API密钥')
    parser.add_argument('--base-url', help='自定义API基础URL')
    
    # 优化参数
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='置信度阈值，低于此值的使用大模型优化')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='API调用最大重试次数')
    
    args = parser.parse_args()
    
    # 构建配置
    config = {
        'enable_llm': args.enable_llm,
        'llm_provider': args.llm_provider,
        'model': args.llm_model,
        'confidence_threshold': args.confidence_threshold,
        'max_retries': args.max_retries
    }
    
    # 设置API密钥
    if args.api_key:
        if args.llm_provider == 'deepseek':
            config['deepseek_api_key'] = args.api_key
        elif args.llm_provider == 'openai':
            config['openai_api_key'] = args.api_key
        else:
            config['api_key'] = args.api_key
    
    if args.base_url:
        config['base_url'] = args.base_url
    
    # 创建转换器
    converter = EnhancedClassicalChineseConverter(config)
    
    if args.batch:
        # 批量处理模式
        output_dir = args.output or args.input
        converter.batch_convert_with_enhancement(args.input, output_dir)
    else:
        # 单个文件处理
        with open(args.input, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # 异步处理
        import asyncio
        enhanced_data = asyncio.run(converter.convert_text_to_json_enhanced(text_content))
        
        output_file = args.output or f"{os.path.splitext(args.input)[0]}_enhanced.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
        
        print(f"增强转换完成: {len(enhanced_data)} 个词条 -> {output_file}")


if __name__ == "__main__":
    main()