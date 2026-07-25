# 项目名称
企业级RAG知识库问答系统

## 📖 项目简介
这是一个面向企业的，有解析PDF能力、处理表格能力的RAG知识库问答系统，混合搜索、重排序、路由（意图识别）、多模态输入、CoT结构化输出。

## ✨ 功能特性
- 支持PDF解析、纯文本解析
- 支持网页对话式询问
- Faiss引擎实现毫秒级的计算速度，降低响应时间

## 🚀 快速开始
### 前置条件
- 环境变量中有DASHSCOPE_API_KEY
- 按照requirement安装环境

### 安装步骤
```bash
git clone https://github.com/RivanLeon/My-Enterprise-RAG-Knowledge-Base-System.git
cd My-Enterprise-RAG-Knowledge-Base-System
pip install -r requirements.txt
```

### 运行结果示例
<img width="1314" height="968" alt="1" src="https://github.com/user-attachments/assets/496e993f-615e-4803-8d81-02b865645ed9" />

### 原理图
<img width="999" height="517" alt="成品1" src="https://github.com/user-attachments/assets/ce439f23-881b-48a6-bda2-a5c91d6847e9" />


### 技术栈
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![DashScope](https://img.shields.io/badge/DashScope-Qwen-FF6A00?logo=alibabacloud&logoColor=white)
![text-embedding-v1](https://img.shields.io/badge/Embedding-text--embedding--v1-00A67E?logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-1488C6?logo=meta&logoColor=white)
![Rerank](https://img.shields.io/badge/Rerank-LLM%20Reranking-7B68EE?logo=huggingface&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Text%20Splitter-1C3C3C?logo=langchain&logoColor=white)
![MinerU](https://img.shields.io/badge/MinerU-PDF%20Parsing-2E8B57?logo=adobeacrobatreader&logoColor=white)
