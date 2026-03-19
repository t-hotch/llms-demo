"""ReAct agent chatbot using LangChain's modern agent framework.

This demo shows how LLMs can use tools through LangChain's built-in agent support.
The agent can solve multi-step problems by automatically selecting and using tools.

Available tools:
- calculator: Arithmetic operations
- get_current_date: Returns today's date
- days_between: Calculates days between two dates

Example questions to try:
- "How many days until Christmas from today?"
- "Calculate 15% tip on a $47.50 bill"
- "I was born on March 15, 1990. How old am I in days?"
- "What's 25% of 360, divided by 3?"
- "How many weeks between today and New Year's Day 2027?"

Usage:
    python demos/langchain_patterns/react_agent_chatbot.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# Add src directory to path for tool imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our tools
from tools import calculator, get_current_date, days_between

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---

temperature = 0.7

# --- Initialize Ollama backend ---

ollama_model = 'qwen2.5:3b'
ollama_client = ChatOllama(model=ollama_model, temperature=temperature)

# --- Initialize llama.cpp backend ---

llamacpp_server = os.environ.get('PERDRIZET_URL', 'localhost:8502')

if llamacpp_server.startswith('localhost') or llamacpp_server.startswith('127.'):
    llamacpp_api_key = os.environ.get('LLAMA_API_KEY', 'dummy')
    llamacpp_base_url = f'http://{llamacpp_server}/v1'
else:
    llamacpp_api_key = os.environ.get('PERDRIZET_API_KEY')
    llamacpp_base_url = f'https://{llamacpp_server}/v1'

llamacpp_client = ChatOpenAI(
    base_url=llamacpp_base_url,
    api_key=llamacpp_api_key,
    timeout=120.0,
    model='gpt-oss-20b',
    temperature=temperature
)

llamacpp_model = 'gpt-oss-20b'

# --- Tools list ---
TOOLS = [calculator, get_current_date, days_between]

# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful assistant that can use tools to answer questions.

Think step by step and use tools when you need to:
- Perform calculations
- Get the current date
- Calculate days between dates

Always explain your reasoning before providing the final answer."""


def create_agent_for_backend(backend: str):
    """Create a LangChain agent for the specified backend.
    
    Args:
        backend: Either 'Ollama' or 'llama.cpp'
    
    Returns:
        Compiled agent graph
    """
    llm = ollama_client if backend == 'Ollama' else llamacpp_client
    
    # Create the agent using LangChain's API
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        debug=True  # Print intermediate steps to console
    )
    
    return agent


def format_messages(messages: List[Dict[str, Any]]) -> str:
    """Format agent messages for display in reasoning pane.
    
    Args:
        messages: List of message objects from the agent
    
    Returns:
        Formatted markdown string
    """

    if not messages:
        return "*No reasoning steps*"
    
    formatted = []
    tool_calls = []
    
    for msg in messages:

        # Handle both dict and Message object formats
        if hasattr(msg, 'type'):
            msg_type = msg.type
            content = msg.content if hasattr(msg, 'content') else ''

        else:
            msg_type = msg.get('type', '')
            content = msg.get('content', '')
        
        # Track tool calls
        if msg_type == 'ai':

            # Check for tool_calls attribute or key
            tc = getattr(msg, 'tool_calls', None) or (msg.get('tool_calls') if isinstance(msg, dict) else None)

            if tc:
                for tool_call in tc:
                    if hasattr(tool_call, 'get'):
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})

                    else:
                        tool_name = getattr(tool_call, 'name', 'unknown')
                        tool_args = getattr(tool_call, 'args', {})

                    tool_calls.append(f"{tool_name}({tool_args})")
        
        # Track tool responses
        if msg_type == 'tool':
            tool_name = getattr(msg, 'name', None) or (msg.get('name', 'unknown') if isinstance(msg, dict) else 'unknown')
            formatted.append(f"**Tool: {tool_name}**")
            formatted.append(f"*Result:* {content}")
            formatted.append("")
    
    if tool_calls:
        formatted.insert(0, f"**Tools used:** {', '.join(tool_calls)}")
        formatted.insert(1, "")
    
    return "\n".join(formatted) if formatted else "*Agent completed without using tools*"


def respond(message, history, backend):
    """Sends message to ReAct agent, gets response with reasoning steps.
    
    Args:
        message: User's current message
        history: Chat history from Gradio (ignored - agent handles its own state)
        backend: Either 'Ollama' or 'llama.cpp'
    
    Returns:
        Tuple of (response_text, reasoning_steps_text)
    """
    try:
        # Create agent for this backend
        agent = create_agent_for_backend(backend)
        
        # Run the agent
        result = agent.invoke({"messages": [{"role": "user", "content": message}]})
        
        # Extract the final response
        final_messages = result.get('messages', [])
        
        # Get the last message content (should be an AIMessage)
        if final_messages:

            last_message = final_messages[-1]

            # Handle both dict and Message object formats
            if hasattr(last_message, 'content'):
                answer = last_message.content

            elif isinstance(last_message, dict):
                answer = last_message.get('content', 'No response generated')

            else:
                answer = str(last_message)
        else:
            answer = 'No response generated'
        
        # Format reasoning steps
        reasoning = format_messages(final_messages)
        
        return answer, reasoning
    
    except Exception as e:

        error_msg = (
            f'**Error occurred**\n\n'
            f'{str(e)}\n\n'
            f'**Troubleshooting:**\n'
            f'- Make sure the selected backend is running\n'
            f'- Ollama: `ollama serve`\n'
            f'- llama.cpp: check server at {llamacpp_base_url}\n'
            f'- Try a simpler question'
        )

        import traceback
        print(traceback.format_exc())

        return error_msg, f"*Error: {str(e)}*"


def handle_message(message, history, backend):
    """Process message and update both chat and reasoning displays."""

    if not message.strip():
        return history, history, ""
    
    # Get response and reasoning from agent
    response, reasoning = respond(message, history, backend)
    
    # Update chat history with Gradio 6.0 message format
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]
    
    return history, history, reasoning


# --- Build Gradio UI ---

with gr.Blocks(title='ReAct Agent Demo') as demo:
    
    gr.Markdown("""
    # ReAct Agent Demo
    
    This chatbot uses LangChain's built-in agent framework to solve multi-step problems.
    Watch how it automatically selects and uses tools to get information.
    
    **Available tools:** Calculator, Current Date, Days Between Dates
    """)
    
    # Backend selector
    with gr.Row():

        backend_selector = gr.Radio(
            choices=['Ollama', 'llama.cpp'],
            value='Ollama',
            label='Model Backend',
            info=f'Ollama: {ollama_model} | llama.cpp: {llamacpp_model} @ {llamacpp_base_url}'
        )
    
    # Example questions
    gr.Markdown("""
    **Try these questions:**
    - How many days until Christmas from today?
    - Calculate 15% tip on a $47.50 bill
    - I was born on March 15, 1990. How old am I in days?
    - What's 25% of 360, divided by 3?
    """)
    
    # Two-column layout: chat interface + reasoning trace
    with gr.Row():
        with gr.Column(scale=1):
    
            gr.Markdown("### Chat")
            chatbot_display = gr.Chatbot(height=550)
    
            msg_input = gr.Textbox(
                label="Your question",
                placeholder="Ask a math or date question...",
                lines=2
            )

            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear")
        
        with gr.Column(scale=1):
    
            gr.Markdown("### Reasoning Process")

            reasoning_display = gr.Markdown(
                value="*Reasoning steps will appear here*"
            )
    
    # Connect submit button
    submit_btn.click(
        fn=handle_message,
        inputs=[msg_input, chatbot_display, backend_selector],
        outputs=[chatbot_display, chatbot_display, reasoning_display]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )
    
    # Clear button
    clear_btn.click(
        lambda: ([], "*Reasoning steps will appear here*"),
        outputs=[chatbot_display, reasoning_display]
    )


# Launch the Gradio app
if __name__ == '__main__':
    demo.launch()
