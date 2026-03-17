# LLM chatbots demo

[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://gperdrizet.github.io/llms-demo/)

## Introduction

This repository provides hands-on examples and learning resources for working with large language models (LLMs) in local development environments. It demonstrates practical approaches to prompt engineering, model deployment, and application development.

### Topics covered

**Inference and deployment:**
- Local inference servers (Ollama, llama.cpp)
- CPU/GPU memory management for large models
- OpenAI-compatible APIs
- Model quantization (GGUF format)

**Development tools and libraries:**
- HuggingFace Transformers for direct model loading
- LangChain for structured LLM applications
- Gradio for rapid UI prototyping
- Multi-backend architecture patterns

**Prompting techniques:**
- Zero-shot and few-shot learning
- Chain-of-thought reasoning
- System prompt design
- Iterative prompt refinement

### Resources included

**Demos** (5 chatbot implementations):
- Terminal chatbot with Ollama + LangChain
- Terminal chatbot with llama.cpp + OpenAI client
- Web UI chatbot with Gradio (multi-backend)
- Direct model loading with HuggingFace Transformers
- ReAct agent with tool use (2 versions: built-in framework + manual implementation)

**Slides** (4 lessons):
- State of the art in generative AI
- LLM deployment strategies
- Prompting fundamentals
- Advanced prompting techniques

**Activities** (3 hands-on exercises):
- Word problems with chain-of-thought reasoning
- Text summarization with document chunking
- Extending the ReAct agent with custom tools

**Models** (3 open-source LLMs):
- GPT-OSS-120B (120B MoE, reasoning-optimized)
- GPT-OSS-20B (21B MoE, fast inference)
- Qwen3.5-35B-A3B (35B MoE with vision)

---

## Documentation

**Complete documentation:** [https://gperdrizet.github.io/llms-demo/](https://gperdrizet.github.io/llms-demo/)

The documentation covers:
- Setup and installation
- Demo usage and concepts
- Inference server configuration
- Library reference with code examples
- Model specifications and serving commands
- Systemd deployment for production use
- Slide and activity guides

---

## Quickstart

### 1. Fork and clone

1. Click **Fork** in the top-right corner of this repo on GitHub to create your own copy.
2. Clone your fork:

   ```bash
   git clone https://github.com/<your-username>/llms-demo.git
   ```

### 2. Open in a dev container

1. Open the cloned folder in VS Code.
2. When prompted **"Reopen in Container"**, click it - or run the command **Dev Containers: Reopen in Container** from the Command Palette (`Ctrl+Shift+P`).
3. VS Code will build and start the container. This takes a few minutes the first time.

### 3. What happens during container startup

The dev container is based on the `gperdrizet/deeplearning-gpu` image (NVIDIA GPU-enabled). On first creation, the `postCreateCommand` runs automatically and does the following:

| Step | What it does |
|------|-------------|
| `mkdir -p models/hugging_face && mkdir -p models/ollama` | Creates local directories for model storage |
| `pip install -r requirements.txt` | Installs Python dependencies: **gradio**, **huggingface-hub**, **langchain-ollama**, **openai**, **python-dotenv**, **torch**, **transformers** |
| `bash .devcontainer/install_ollama.sh` | Downloads and installs the Ollama CLI |

The container also pre-configures the following:

| Setting | Detail |
|---------|--------|
| **GPU access** | All host GPUs are passed through (`--gpus all`) |
| **Python interpreter** | `/usr/bin/python` is set as the default |
| **`HF_HOME`** | Points to `models/hugging_face` so Hugging Face downloads stay in the repo |
| **`OLLAMA_MODELS`** | Points to `models/ollama` so Ollama downloads stay in the repo |
| **Port 7860** | Forwarded automatically for Gradio web UIs |
| **VS Code extensions** | Python, Jupyter, Code Spell Checker, and Marp (slide viewer) are installed |

Once the container is ready you can start running the demos - no extra setup needed.

---

## Running the demos

See the [Demos documentation](https://gperdrizet.github.io/llms-demo/demos.html) for detailed instructions on running each chatbot, including:
- Concepts covered in each demo
- Tools and libraries used
- Step-by-step setup and execution

**Quick example** - Ollama chatbot:
```bash
# 1. Start the Ollama server
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the chatbot
python src/ollama_chatbot.py
```

For complete instructions on all four demos, visit the [documentation](https://gperdrizet.github.io/llms-demo/).


