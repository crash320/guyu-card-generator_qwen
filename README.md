# 古汉语常用词 Anki 卡片自动化生成系统

一个基于 Python 的古汉语常用词 Anki 卡片自动化生成系统，支持大模型增强的文本解析和多种输出格式。

## 功能特性

### 🎯 核心功能
- **智能文本解析**: 自动解析古汉语文本结构，提取词条、义项、例句等信息
- **大模型增强**: 集成 OpenAI GPT、Claude 等大模型优化解析结果
- **多格式输出**: 支持 CSV 和 .apkg (Anki 包) 格式输出
- **批量处理**: 支持批量转换文本文件
- **质量验证**: 自动生成转换质量报告

### 📊 卡片类型
- **总览卡**: 快速建立语义地图，包含所有义项摘要
- **细节卡**: 深度强化区分义项，保留全部例句

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基本使用

#### 1. 文本到 JSON 转换
```bash
# 基本转换（不使用大模型）
python enhanced_text_to_json.py 古汉语常用词样例.txt -o vocabulary.json

# 启用大模型优化
python enhanced_text_to_json.py 古汉语常用词样例.txt -o enhanced_vocabulary.json \
  --enable-llm --llm-provider openai --api-key $OPENAI_API_KEY
```

#### 2. 批量处理
```bash
# 批量转换目录中的所有文本文件
python enhanced_text_to_json.py ./text_files/ -o ./output/ --batch --enable-llm
```

#### 3. 测试转换器
```bash
python test_enhanced_converter.py
```

## 配置选项

### 大模型配置
```bash
# OpenAI GPT
--enable-llm --llm-provider openai --api-key $OPENAI_API_KEY --llm-model gpt-4

# Anthropic Claude
--enable-llm --llm-provider anthropic --api-key $ANTHROPIC_API_KEY --llm-model claude-3-sonnet-20240229

# 自定义 API
--enable-llm --llm-provider custom --base-url http://localhost:8080 --api-key $CUSTOM_API_KEY
```

### 优化参数
```bash
--confidence-threshold 0.7    # 置信度阈值（0-1）
--max-retries 3               # API 调用重试次数
```

## 输入格式

### 文本格式示例
```
1.【言】
（一）动词。说话，说。《论语·乡党》："食不語，寢不～。"《左传·成公二年》："豈敢～病？"引申为[谈问题，对某事表示意见]。《左传·僖公四年》："楚子使與師～曰。"《战国策·赵策三》："勝也何敢～事？"（勝：趙勝。平原君自稱。）《史记·廉颇蔺相如列传》："趙括自少時學兵法、～兵事。"
（二）名词。话，言论。《论语·公冶长》："聽其～而觀其行。"引申为[一句话为一言]。《论语·为政》："詩三百，一～以蔽之，曰'思無邪'。"（詩三百：詩經三百篇。）又[一个字为一言]。《论语·卫灵公》："子貢問曰：'有一～而可以終身行之者乎？'子曰：'其"恕"乎！'"《史记·老子韩非列传》："於是老子乃著書上下篇，言道德之意，五千餘～。"又如"五～詩""七～詩"。
```

### 输出 JSON 格式
```json
{
  "word": "言",
  "pinyin": null,
  "meanings": [
    {
      "index": "一",
      "part_of_speech": "动词",
      "definition": "说话，说",
      "examples": [
        "《论语·乡党》：'食不語，寢不～。'",
        "《左传·成公二年》：'豈敢～病？'"
      ],
      "extended_meanings": [
        "谈问题，对某事表示意见"
      ]
    }
  ],
  "comparisons": []
}
```

## 项目结构

```
guyu-card-generator_qwen/
├── enhanced_text_to_json.py      # 大模型增强的转换器
├── test_enhanced_converter.py    # 测试脚本
├── requirements.txt              # 依赖包
├── README.md                     # 项目说明
├── 古汉语常用词样例.md           # 需求文档和样例
├── 工程实践指南.md               # 降低完美主义负担的实践
├── Python批量脚本设计.md         # 批量处理脚本设计
├── 古汉语数据处理脚本设计.md     # 数据处理脚本设计
├── 文本到JSON转换器设计.md       # 基础转换器设计
└── 大模型增强的文本转换器设计.md # 大模型增强设计
```

## 技术架构

### 解析流程
1. **文本预处理**: 清理和统一文本格式
2. **词条分割**: 按词条编号分割文本
3. **结构解析**: 解析义项、例句、引申义
4. **大模型优化**: 对低置信度条目进行优化
5. **数据验证**: 检查解析完整性
6. **JSON输出**: 生成结构化数据

### 大模型集成
- **多提供商支持**: OpenAI, Anthropic, 自定义 API
- **智能优化**: 基于置信度的选择性优化
- **错误处理**: 重试机制和降级处理
- **质量评估**: 自动生成质量报告

## 使用场景

### 🎓 学习场景
- 古汉语词汇记忆
- 文言文阅读理解
- 汉语语言学学习

### 🔧 技术场景
- 自然语言处理研究
- 文本数据挖掘
- 教育技术开发

## 开发指南

### 环境设置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行测试
```bash
python test_enhanced_converter.py
```

### 代码规范
```bash
# 代码格式化
black enhanced_text_to_json.py test_enhanced_converter.py

# 代码检查
flake8 enhanced_text_to_json.py test_enhanced_converter.py
```

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 使用大模型功能需要配置相应的 API 密钥，请确保遵守各服务商的使用条款。