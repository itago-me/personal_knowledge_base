



# 个人知识库问答系统

## 项目简介
这是一个基于 RAG（检索增强生成）技术的个人知识库问答系统。你可以上传自己的文档（PDF、Markdown、TXT等），然后通过自然语言提问，系统实现了多轮会话保存,系统会从你的文档中找到相关内容并生成回答。

## 项目目录

├── api.py
├── config
│   └── config.yaml
│       ├── 1.txt
│       ├── async_python.txt
│       ├── text.md
│       └── text.txt
├── README.md
├── requirement.txt
├── src
│   ├── document_loader.py
│   ├── embedding.py
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── text_splitter.py
│   └── vector_store.py
├── templates
│   └── index.html

## 功能特点
- 支持多种文档格式
- 文本分块
- 本地向量存储，保护隐私
- 模型生成式回答
- 可选用 OpenAI API 或本地模型
- 简单易用的 Web 界面（开发中）

## 技术栈
- Python 3.10
- fastapi 
- LangChain
- ChromaDB
- Sentence-Transformers
- Streamlit

## 快速开始
### 1. 环境配置
```bash
# 创建 conda 环境
conda create -n kb_project python=3.10.3
conda activate kb_project

# 安装依赖
pip install -r requirements_frozen.txt

```
