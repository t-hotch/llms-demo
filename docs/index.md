# llms-demo

[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://gperdrizet.github.io/llms-demo/)

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

- **Terminal chatbots**: Connect to Ollama or llama.cpp servers using LangChain and OpenAI clients
- **Web UI chatbot**: Multi-backend Gradio interface with customizable system prompts
- **Direct model loading**: HuggingFace Transformers without an inference server

### Infrastructure

- **Inference servers**: [Ollama](inference_servers.md) (lightweight), [llama.cpp](inference_servers.md) (high-performance MoE)
- **Libraries**: [Transformers](libraries.md), [LangChain](libraries.md), [Gradio](libraries.md)
- **Models**: [GPT-OSS-120B](models.md) (120B MoE), [GPT-OSS-20B](models.md) (21B), [Qwen3.5-35B-A3B](models.md) (35B MoE with vision)

### Get started

See the [Quickstart](quickstart.md) guide for installation and setup, then explore the [Demos](demos.md) to learn about different inference approaches.

---

## Links

- [GitHub repository](https://github.com/gperdrizet/llms-demo)
- [Docker image](https://hub.docker.com/repository/docker/gperdrizet/deeplearning-gpu/general)

