"""Manual ReAct agent implementation without LangChain's agent framework.

This demo shows how to implement the ReAct pattern from scratch with:
- Manual prompting for Thought/Action/Observation cycles
- Regex-based parsing of LLM outputs
- Explicit tool execution loop
- Custom reasoning trace formatting

This is educational to understand how agent frameworks work internally.
Compare this with react_agent_chatbot.py which uses LangChain's built-in agent.

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
    python src/react_agent_chatbot_manual.py
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Add src directory to path for tool imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our tools
from tools import calculator, get_current_date, days_between

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---

temperature = 0.7
MAX_ITERATIONS = 10  # Prevent infinite loops

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

# --- Tool registry ---

TOOLS = {
    'calculator': calculator,
    'get_current_date': get_current_date,
    'days_between': days_between
}

# Build tool descriptions for the prompt
TOOL_DESCRIPTIONS = """
Available tools:

1. calculator(expression: str) -> str
   - Evaluates mathematical expressions
   - Example: calculator("(15 * 0.15)")

2. get_current_date() -> str
   - Returns today's date in YYYY-MM-DD format
   - Example: get_current_date()

3. days_between(start_date: str, end_date: str) -> str
   - Calculates days between two dates (format: YYYY-MM-DD)
   - Example: days_between("2024-01-01", "2024-12-25")
"""

# System prompt that teaches the ReAct pattern
SYSTEM_PROMPT = f"""You are a helpful assistant that solves problems step by step using the ReAct pattern.

{TOOL_DESCRIPTIONS}

CRITICAL: You MUST follow this EXACT format for every response:

Thought: [Your reasoning about what to do next]
Action: [tool_name(arguments)]

After you write "Action:", wait for the system to provide "Observation:" with the result.
Then continue with another "Thought:" and either another "Action:" or final "Answer:".

When you have enough information to answer:
Thought: [Final reasoning]
Answer: [Your final response to the user]

FORMATTING RULES (MUST FOLLOW):
1. ALWAYS start your response with "Thought:" 
2. If you need a tool, write "Action:" on the next line with the function call
3. Use Python function call syntax: tool_name("arg1", "arg2")
4. Do NOT use JSON format for arguments
5. Do NOT output raw function arguments without the Action: prefix
6. Wait for the Observation before continuing
7. When ready to answer, use "Answer:" instead of "Action:"

CORRECT Example 1 (calculation):
Thought: I need to calculate 15% of 100
Action: calculator("100 * 0.15")
[System provides: Observation: 15.0]
Thought: I have the answer
Answer: 15% of 100 is 15

CORRECT Example 2 (date calculation):
Thought: I need to know today's date first
Action: get_current_date()
[System provides: Observation: 2026-03-17]
Thought: Now I can calculate days until Christmas (2026-12-25)
Action: days_between("2026-03-17", "2026-12-25")
[System provides: Observation: 283]
Thought: I have the answer
Answer: There are 283 days until Christmas from today.

INCORRECT Examples (DO NOT DO THIS):
- {{"start_date":"2026-03-17","end_date":"2026-12-25"}}
- days_between("2026-03-17", "2026-12-25")
- Just outputting tool arguments without "Thought:" and "Action:"

Remember: ALWAYS start with "Thought:", then use "Action:" for tool calls.
"""


def parse_action(text: str) -> Optional[Tuple[str, str]]:
    """Parse an action from the LLM's response.
    
    Handles multiple formats:
    - Standard: Action: tool_name(arguments)
    - Bare call: tool_name(arguments) without Action: prefix
    - JSON: Detects tool arguments in JSON format and infers tool name
    
    Args:
        text: The LLM's response text
    
    Returns:
        Tuple of (tool_name, arguments_str) or None if no action found
    """
    # Pattern 1: Standard format - Action: tool_name(arguments)
    pattern1 = r'Action:\s*(\w+)\s*\((.*?)\)'
    match = re.search(pattern1, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        return (tool_name, args_str)
    
    # Pattern 2: Bare tool call without "Action:" prefix
    # Look for known tool names followed by parentheses
    for tool_name in TOOLS.keys():
        pattern2 = rf'\b{tool_name}\s*\((.*?)\)'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            args_str = match.group(1).strip()
            return (tool_name, args_str)
    
    # Pattern 3: JSON format with start_date/end_date (indicates days_between call)
    json_pattern = r'\{[^}]*"start_date"[^}]*"end_date"[^}]*\}'
    json_match = re.search(json_pattern, text)
    if json_match:
        try:
            json_obj = json.loads(json_match.group(0))
            if 'start_date' in json_obj and 'end_date' in json_obj:
                # Convert to function call format
                args_str = f'"{json_obj["start_date"]}", "{json_obj["end_date"]}"'
                return ('days_between', args_str)
        except:
            pass
    
    return None


def parse_answer(text: str) -> Optional[str]:
    """Parse the final answer from the LLM's response.
    
    Args:
        text: The LLM's response text
    
    Returns:
        The answer text or None if not found
    """
    # Look for Answer: <text>
    pattern = r'Answer:\s*(.+?)(?:\n|$)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return None


def execute_tool(tool_name: str, args_str: str) -> str:
    """Execute a tool with the given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        args_str: String containing the arguments
    
    Returns:
        Tool execution result or error message
    """
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'. Available tools: {', '.join(TOOLS.keys())}"
    
    tool_func = TOOLS[tool_name]
    
    try:
        # Parse arguments - handle both single strings and multiple args
        # For safety, we use eval in a restricted way
        if args_str:

            # Split by comma for multiple arguments, strip quotes/whitespace
            args = []

            for arg in args_str.split(','):

                arg = arg.strip()

                # Remove surrounding quotes if present
                if (arg.startswith('"') and arg.endswith('"')) or \
                   (arg.startswith("'") and arg.endswith("'")):
                    arg = arg[1:-1]

                args.append(arg)
            
            # Call the tool with parsed arguments
            result = tool_func.func(*args) if hasattr(tool_func, 'func') else tool_func(*args)
        else:
            # No arguments
            result = tool_func.func() if hasattr(tool_func, 'func') else tool_func()
        
        return str(result)
    
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def run_react_loop(question: str, llm: Any) -> Tuple[str, List[str]]:
    """Run the ReAct loop manually.
    
    Args:
        question: User's question
        llm: LangChain LLM client (ChatOllama or ChatOpenAI)
    
    Returns:
        Tuple of (final_answer, reasoning_steps)
    """

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]
    
    reasoning_steps = []
    reasoning_steps.append(f"**Question:** {question}\n")
    
    for iteration in range(MAX_ITERATIONS):

        # Get LLM response
        response = llm.invoke(messages)
        response_text = response.content
        
        # Check for action first (priority over answer detection)
        action = parse_action(response_text)

        if action:
            tool_name, args_str = action
            
            # Execute the tool
            observation = execute_tool(tool_name, args_str)
            
            # Format this iteration nicely
            reasoning_steps.append(f"**Iteration {iteration + 1}:**")
            
            # Extract and show "Thought:" if present
            thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|Action:|\nAnswer:|Answer:|$)', response_text, re.IGNORECASE | re.DOTALL)
            if thought_match:
                reasoning_steps.append(f"Thought: {thought_match.group(1).strip()}")
            
            # Show the action in standardized format
            reasoning_steps.append(f"Action: {tool_name}({args_str})")
            reasoning_steps.append(f"Observation: {observation}")
            reasoning_steps.append("")
            
            # Add observation to conversation
            messages.append(AIMessage(content=response_text))
            messages.append(HumanMessage(content=f"Observation: {observation}"))
            continue
        
        # Check for final answer (only if no action was found)
        answer = parse_answer(response_text)

        if answer:
            reasoning_steps.append(f"**Iteration {iteration + 1}:**")
            
            # Extract and show "Thought:" if present before the answer
            thought_match = re.search(r'Thought:\s*(.+?)(?=\nAnswer:|Answer:|$)', response_text, re.IGNORECASE | re.DOTALL)
            if thought_match:
                thought_text = thought_match.group(1).strip()
                # Make sure we don't capture the answer itself
                if not thought_text.startswith(answer):
                    reasoning_steps.append(f"Thought: {thought_text}")
            
            # Show the answer
            reasoning_steps.append(f"Answer: {answer}")
            reasoning_steps.append("")
            return answer, reasoning_steps

        # No action or answer pattern found - treat as final response
        reasoning_steps.append(f"**Iteration {iteration + 1}:**")
        reasoning_steps.append(response_text)
        reasoning_steps.append("")
        
        # Extract text after "Thought:" (but not including "Action:")
        # Use a non-greedy match that stops at "Action:" or end of string
        thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', response_text, re.IGNORECASE | re.DOTALL)

        if thought_match:
            return thought_match.group(1).strip(), reasoning_steps

        else:
            return response_text, reasoning_steps
    
    # Max iterations reached
    reasoning_steps.append("**Error:** Maximum iterations reached without finding answer")
    return "I apologize, but I couldn't solve this problem within the iteration limit.", reasoning_steps


def respond(message: str, history: List, backend: str) -> Tuple[str, str]:
    """Process message through manual ReAct agent.
    
    Args:
        message: User's current message
        history: Chat history from Gradio (ignored - we maintain our own state)
        backend: Either 'Ollama' or 'llama.cpp'
    
    Returns:
        Tuple of (response_text, reasoning_steps_text)
    """

    try:
        # Select backend
        llm = ollama_client if backend == 'Ollama' else llamacpp_client
        
        # Run the manual ReAct loop
        answer, reasoning_steps = run_react_loop(message, llm)
        
        # Format reasoning steps
        reasoning_text = "\n".join(reasoning_steps)
        
        return answer, reasoning_text
    
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


def handle_message(message: str, history: List, backend: str) -> Tuple[List, List, str]:
    """Process message and update both chat and reasoning displays.
    
    Args:
        message: User's message
        history: Current chat history
        backend: Selected backend
    
    Returns:
        Tuple of (updated_history, updated_history, reasoning_text)
    """

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

with gr.Blocks(title='Manual ReAct Agent Demo') as demo:
    
    gr.Markdown("""
    # Manual ReAct Agent Demo
    
    This chatbot implements the ReAct pattern **manually** without using LangChain's agent framework.
    Watch how it explicitly follows the Thought → Action → Observation loop.
    
    **Available tools:** Calculator, Current Date, Days Between Dates
    
    **Compare with:** `react_agent_chatbot.py` (uses LangChain's built-in agent)
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
