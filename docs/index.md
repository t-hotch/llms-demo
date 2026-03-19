# Home

[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://gperdrizet.github.io/llms-demo/)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?logo=gradio&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)
![llama.cpp](https://img.shields.io/badge/llama.cpp-GGUF-green)
![PostgreSQL](https://img.shields.io/badge/pgvector-PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-E8572A)

Example code demonstrating local LLM inference with various backends and libraries.

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
slides
demos
activities
inference_servers
libraries
models
systemd-deployment
```

## Overview

This repository contains chatbot demos and hands-on activities for learning prompt engineering and local LLM deployment.

### Demos

**Chatbots** (`demos/chatbots/`)
- **HuggingFace chatbot**: Direct model loading with Transformers — no inference server needed
- **Ollama chatbot**: Terminal chatbot using LangChain + a local Ollama server
- **llama.cpp chatbot**: Terminal chatbot using the OpenAI-compatible llama.cpp API
- **Gradio chatbot**: Web UI with switchable Ollama / llama.cpp backends and customizable system prompt

**LangChain patterns** (`demos/langchain_patterns/`)
- **LangChain demo**: Prompt templates, output parsers, LCEL chains, and few-shot learning
- **ReAct agent**: LangChain agent with custom tools and multi-step reasoning (two versions: framework and manual)

**RAG system** (`demos/rag_system/`)
- **RAG demo**: Ingest Wikipedia articles into a pgvector knowledge base and query them with a grounded LLM

### Infrastructure

- **Inference servers**: [Ollama](inference_servers.md) (lightweight), [llama.cpp](inference_servers.md) (high-performance MoE)
- **Libraries**: [Transformers](libraries.md), [LangChain](libraries.md), [Gradio](libraries.md)
- **Models**: [GPT-OSS-120B](models.md) (120B MoE), [GPT-OSS-20B](models.md) (21B), [Qwen3.5-35B-A3B](models.md) (35B MoE with vision) — any GGUF-compatible model can be added by downloading the weights and pointing the server at them

### Get started

See the [Quickstart](quickstart.md) guide for installation and setup, then explore the [Demos](demos.md) to learn about different inference approaches.

---

## Links

- [GitHub repository](https://github.com/gperdrizet/llms-demo)
- [Docker image](https://hub.docker.com/repository/docker/gperdrizet/llms-gpu/general)

