# LLM chatbots demo

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
| `pip install -r requirements.txt` | Installs Python dependencies: **gradio**, **langchain-ollama**, **python-dotenv**, **torch**, **transformers** |
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

## Demos

| File | What it does | Stack |
|------|-------------|-------|
| `src/ollama_chatbot.py` | Terminal chatbot | Ollama + LangChain |
| `src/llamacpp_chatbot.py` | Terminal chatbot using a large MoE model | llama.cpp + OpenAI client |

---

## Demo cheat sheet

Quick reference for the tools and libraries used in the demos:

1. Ollama
2. HuggingFace Transformers
3. LangChain
4. Gradio
5. llama.cpp

### 1. [Ollama](https://ollama.com/)

Local inference server - runs models on your machine behind a REST API.

- [Model library](https://ollama.com/library)
- [Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)
- [API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Start the server (runs on localhost:11434)
ollama serve

# Pull a model
ollama pull qwen2.5:3b

# List downloaded models
ollama list

# Run a model interactively
ollama run qwen2.5:3b

# Remove a model
ollama rm qwen2.5:3b
```

>**Note**: if you are running the demos in this repo via a devcontainer as intended, you do not need to install Ollama. The container environment includes it.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_MODELS` | Directory where models are stored |
| `OLLAMA_HOST` | Server address (default `127.0.0.1:11434`) |

---

### 2. [HuggingFace Transformers](https://huggingface.co/docs/transformers)

Load and run models directly in Python - no server needed.

- [Model Hub](https://huggingface.co/models)
- [Documentation](https://huggingface.co/docs/transformers)
- [GitHub](https://github.com/huggingface/transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer (downloads on first run)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')

# Build a prompt using the chat template
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Hello!'},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Tokenize and generate
inputs = tokenizer(text, return_tensors='pt').to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)

# Decode only the new tokens
new_tokens = output[0][inputs['input_ids'].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

### Key classes

| Class | Purpose |
|-------|---------|
| `AutoTokenizer` | Loads the correct tokenizer for any model |
| `AutoModelForCausalLM` | Loads a causal language model for text generation |
| `apply_chat_template()` | Formats a list of messages into the model's expected prompt format |
| `model.generate()` | Runs inference and returns generated token IDs |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_HOME` | Directory where models/cache are stored |

---

### 3. [LangChain](https://python.langchain.com/)

Framework for building LLM applications. Used here for chat message types and the Ollama integration.

- [Documentation](https://python.langchain.com/docs)
- [langchain-ollama](https://python.langchain.com/docs/integrations/chat/ollama)
- [Other avalible chat integrations](https://docs.langchain.com/oss/python/integrations/chat)

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Create the LLM client (talks to a running Ollama server)
llm = ChatOllama(
    model='qwen2.5:3b',
    temperature=0.7,
)

# Build a conversation as a list of message objects
messages = [
    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content='Hello!'),
]

# Get a response
response = llm.invoke(messages)
print(response.content)

# Append the response to history and continue
messages.append(response)  # response is an AIMessage
messages.append(HumanMessage(content='Follow-up question'))
response = llm.invoke(messages)
```

### Key classes

| Class | Purpose |
|-------|---------|
| `ChatOllama` | LLM client that connects to Ollama |
| `SystemMessage` | Sets the model's behavior/persona |
| `HumanMessage` | A user message |
| `AIMessage` | A model response |

---

### 4. [Gradio](https://www.gradio.app)

Build a web UI for a chatbot with a few lines of code.

- [Documentation](https://www.gradio.app/docs)
- [GitHub](https://github.com/gradio-app/gradio)
- [ChatInterface guide](https://www.gradio.app/guides/creating-a-chatbot-fast)

```python
import gradio as gr

def respond(message, history):
    # message: str - the latest user input
    # history: list of [user, assistant] pairs
    return 'Hello!'  # return the assistant's reply as a string

demo = gr.ChatInterface(
    fn=respond,
    title='My Chatbot',
)

demo.launch()
```

### `gr.ChatInterface`

| Parameter | Purpose |
|-----------|---------|
| `fn` | Response function `(message, history) -> str` |
| `title` | Title shown in the web page |

### `demo.launch()`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `server_name` | `127.0.0.1` | Bind address (`0.0.0.0` for external access) |
| `server_port` | `7860` | Port number |
| `share` | `False` | Create a public Gradio link |

---

### 5. [llama.cpp](https://github.com/ggml-org/llama.cpp)

High-performance C/C++ inference engine. Runs GGUF-quantized models and can split MoE layers across CPU and GPU, making it possible to serve large models (100B+) on consumer hardware.

- [GitHub](https://github.com/ggml-org/llama.cpp)
- [GGUF model format](https://huggingface.co/docs/hub/gguf)
- [Server documentation](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)

```bash
# Build from source with CUDA support (compiles for your GPU automatically)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

> **Note**: The build compiles CUDA kernels for the GPU(s) detected on your machine.
> This takes several minutes but only needs to be done once. If you change GPUs, rebuild.

```bash
# Start the server with CPU/GPU MoE split
./build/bin/llama-server \
    -m model.gguf \
    --n-gpu-layers 999 \
    --n-cpu-moe 36 \
    -c 0 -fa \
    --jinja --reasoning-format none \
    --host 0.0.0.0 --port 8502 --api-key "$GPT_API_KEY"
```

The server exposes an **OpenAI-compatible API**, so any OpenAI client library can connect to it.

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8502/v1',
    api_key='your-api-key',
)

response = client.chat.completions.create(
    model='model-name',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello!'},
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Key server flags

| Flag | Purpose |
|------|----------|
| `-m` | Path to the GGUF model file |
| `--n-gpu-layers` | Number of layers to offload to GPU (`999` = all non-MoE layers) |
| `--n-cpu-moe` | Number of MoE blocks to keep on CPU (e.g. `36` = all MoE on CPU) |
| `-c` | Context length (`0` = model maximum) |
| `-fa` | Enable flash attention |
| `--host` / `--port` | Server bind address and port |
| `--api-key` | API key for authenticating requests |

### CPU/GPU MoE split explained

Mixture-of-Experts models have two types of layers: **attention layers** (small, benefit from GPU) and **MoE/expert layers** (large, run well on CPU). The `--n-cpu-moe` flag controls how many MoE blocks stay on CPU:

| Config | VRAM usage | Generation speed |
|--------|-----------|------------------|
| `--n-cpu-moe 36` (all MoE on CPU) | ~5–8 GB | ~18–25 tok/s |
| `--n-cpu-moe 28` (8 MoE on GPU) | ~22 GB | ~25–31 tok/s |

This makes it possible to run a 120B parameter model with as little as 8 GB of VRAM and 64 GB of system RAM.

### Environment variables

| Variable | Purpose |
|----------|----------|
| `GPT_API_KEY` | API key for llama-server authentication |

---

### Model: openai/gpt-oss-120b

A 120B parameter Mixture-of-Experts (MoE) model released by OpenAI. We use the mxfp4-quantized [GGUF version](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) published by [ggml-org](https://github.com/ggml-org) — the organization behind [llama.cpp](https://github.com/ggml-org/llama.cpp), the [GGML](https://github.com/ggml-org/ggml) tensor library, and the [GGUF](https://huggingface.co/docs/hub/gguf) model format.

- [Model card (OpenAI)](https://huggingface.co/openai/gpt-oss-120b)
- [GGUF quantization (ggml-org)](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF)

| Detail | Value |
|--------|-------|
| **Parameters** | 120B (Mixture-of-Experts) |
| **Quantization** | mxfp4 (expert layers), BF16 (attention layers) |
| **Format** | GGUF (3 shards) |
| **Download size** | ~60 GB |
| **Min system RAM** | 64 GB (96 GB recommended) |
| **Min VRAM** | ~5 GB (attention layers only, with `--n-cpu-moe 36`) |

```bash
# Download the GGUF model
python utils/download_gpt_oss_120b.py
```

