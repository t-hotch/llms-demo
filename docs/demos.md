# Demos

This repository includes six chatbot demos that demonstrate different approaches to local LLM inference. Each demo covers specific concepts and tools.

## Demo 1: HuggingFace chatbot

**File:** `demos/chatbots/huggingface_chatbot.py`

**Concepts covered:**
- Direct model loading (no inference server)
- Chat templates and tokenization
- Generation parameters (temperature, max tokens)
- Decoding and response formatting

**Tools used:**
- [HuggingFace Transformers](libraries.md) - Model loading and inference
- PyTorch - Underlying tensor operations

**Running the demo:**

```bash
# 1. Run the chatbot (downloads model on first run)
python demos/chatbots/huggingface_chatbot.py

# Note: This loads the model directly into memory (no inference server needed).
# First run will download approximately 6GB of model files to models/hugging_face/
```

## Demo 2: Ollama chatbot

**File:** `demos/chatbots/ollama_chatbot.py`

**Concepts covered:**
- Using a local inference server (Ollama)
- Structured message types (SystemMessage, HumanMessage, AIMessage)
- Conversation history management
- Terminal-based interaction

**Tools used:**
- [Ollama](inference_servers.md) - Local inference server
- [LangChain](libraries.md) - LLM application framework

**Running the demo:**

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the chatbot
python demos/chatbots/ollama_chatbot.py
```

## Demo 3: llama.cpp chatbot

**File:** `demos/chatbots/llamacpp_chatbot.py`

**Concepts covered:**
- Running large MoE models (120B+ parameters) on consumer hardware
- CPU/GPU memory split for expert layers
- OpenAI-compatible API usage
- Remote vs. local inference servers

**Tools used:**
- [llama.cpp](inference_servers.md) - High-performance C++ inference engine
- OpenAI Python client - Standard API interface

**Running the demo:**

You have two choices: use the hosted model at `gpt.perdrizet.org`, or run llama.cpp locally.

**Option 1: Using the remote server (recommended for quick start)**

```bash
# 1. Create a .env file with your API credentials
cp .env.example .env

# 2. Edit .env and set:
#    PERDRIZET_URL=gpt.perdrizet.org
#    PERDRIZET_API_KEY=your-api-key-here

# 3. Run the chatbot
python demos/chatbots/llamacpp_chatbot.py
```

**Option 2: Running llama.cpp locally**

```bash
# 1. Download a GGUF model (e.g., GPT-OSS-20B)
python utils/download_gpt_oss_20b.py

# 2. Build llama.cpp with CUDA support
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd ..

# 3. Start the llama-server (see model-specific commands in the Models section)
llama.cpp/build/bin/llama-server -m <model.gguf> <flags...>

# 4. Run the chatbot (in another terminal)
python demos/chatbots/llamacpp_chatbot.py
```

> **Note**: For localhost, the defaults work automatically (localhost:8502 with "dummy" API key). For remote servers, configure `PERDRIZET_URL` and `PERDRIZET_API_KEY` in your `.env` file.

## Demo 4: Gradio chatbot

**File:** `demos/chatbots/gradio_chatbot.py`

**Concepts covered:**
- Web-based chat interfaces
- Multi-backend architecture (switching between Ollama/llama.cpp)
- System prompt customization
- Error handling and user feedback

**Tools used:**
- [Gradio](libraries.md) - Rapid UI prototyping
- [LangChain](libraries.md) - LLM orchestration
- [Ollama](inference_servers.md) - Default backend

**Running the demo:**

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the Gradio chatbot
python demos/chatbots/gradio_chatbot.py

# 4. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

## Demo 6: ReAct agent chatbot

**Files:** 
- `demos/langchain_patterns/react_agent_chatbot.py` - Uses LangChain's agent framework
- `demos/langchain_patterns/react_agent_chatbot_manual.py` - Manual implementation from scratch

**Concepts covered:**
- ReAct (Reasoning + Acting) agent pattern
- Multi-step reasoning with tool use
- Tool selection and execution
- Agent iteration loops and error handling
- Comparing high-level frameworks vs. manual implementation

**Tools used:**
- [LangChain](libraries.md) - Agent framework and tool integration
- [Ollama](inference_servers.md) or [llama.cpp](inference_servers.md) - Backend LLM
- [Gradio](libraries.md) - Web interface with reasoning visualization

**Two versions available:**

This demo includes both a production-ready implementation and an educational version that reveals the inner workings:

1. **Built-in agent** (`react_agent_chatbot.py`): Uses LangChain's `create_agent()` API for automatic ReAct pattern handling. This is the recommended approach for real applications.

2. **Manual implementation** (`react_agent_chatbot_manual.py`): A hand-coded ReAct loop with regex parsing that shows exactly what LangChain does behind the scenes. This version demonstrates:
   - How to prompt the LLM to follow the ReAct pattern
   - Parsing LLM responses to extract actions and answers
   - Manual tool execution and observation injection
   - The explicit iteration loop that drives the agent
   
   Use this version to understand the mechanics of agent frameworks before relying on them.

**Running the demo:**

**Version 1: Built-in agent (recommended for beginners)**

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the ReAct agent chatbot
python demos/langchain_patterns/react_agent_chatbot.py

# 4. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

**Version 2: Manual implementation (educational)**

```bash
# Same setup as Version 1, but run:
python demos/langchain_patterns/react_agent_chatbot_manual.py

# This version shows explicit Thought → Action → Observation cycles
```

**Try these example questions:**
- "How many days until Christmas from today?"
- "Calculate 15% tip on a $47.50 bill"
- "I was born on March 15, 1990. How old am I in days?"
- "What's 25% of 360, divided by 3?"
- "How many weeks between today and New Year's Day 2027?"

**What to observe:**
- Watch the **Reasoning Process** panel (right side) to see how the agent thinks
- Notice when it decides to use tools vs. when it can answer directly
- See the Thought → Action → Observation loop in action
- Try asking multi-step questions that require multiple tool calls
- **Compare both versions**: Run the same question through both demos to see how the manual implementation exposes the mechanics that LangChain handles automatically

## Demo 5: LangChain basics

**File:** `demos/langchain_patterns/langchain_demo.py`

**Concepts covered:**
- Chat models and LLM wrappers
- Prompt templates with variable substitution
- Structured output parsing with Pydantic schemas
- Basic chains and composition with LCEL
- Few-shot learning patterns

**Tools used:**
- [LangChain](libraries.md) - Core framework components
- [Ollama](inference_servers.md) or [llama.cpp](inference_servers.md) - Backend LLM
- [Gradio](libraries.md) - Interactive web interface

**Running the demo:**

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the LangChain demo
python demos/langchain_patterns/langchain_demo.py

# 4. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

**Four interactive examples:**

1. **Simple chain**: Prompt template → LLM → String output
   - Try: "machine learning", "photosynthesis", "blockchain"

2. **Sentiment analysis**: Structured JSON output with Pydantic schema
   - Try: Product reviews, comments, social media posts
   - See how the parser extracts sentiment, confidence, and key phrases

3. **Entity extraction**: Different schemas for different entity types
   - Person: name, age, occupation, location
   - Recipe: name, cuisine, ingredients, difficulty
   - Switch schemas to see how the same chain extracts different information

4. **Few-shot learning**: Style classification with examples
   - The model learns from 4 in-prompt examples
   - Try: Technical, casual, formal, or creative writing styles

**What to observe:**
- **Reusability**: Same chain works for multiple inputs
- **Type safety**: Pydantic schemas ensure structured outputs
- **Composability**: Chains combine prompt, model, and parser seamlessly
- **Format instructions**: See how Pydantic schemas generate parsing guidance
