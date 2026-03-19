# Activity 3: Building LangChain chains

## Objective

Build practical LangChain applications using prompt templates, output parsers, and chains. Create reusable components and compose them into workflows.

## Overview

In this activity, you'll build three increasingly complex LangChain applications:

1. **Template-based translator** - Use prompt templates with variables
2. **Structured data extractor** - Parse JSON output with Pydantic schemas
3. **Multi-step chain** - Compose multiple operations into a pipeline

By the end, you'll understand how to structure LLM applications with LangChain's core abstractions.

## Setup

### 1. Start your LLM backend

You can use either Ollama or llama.cpp:

**Option 1: Ollama**
```bash
# Already running if installed
ollama list  # verify qwen2.5:3b is available
```

**Option 2: llama.cpp server**
```bash
cd llama.cpp
./build/bin/llama-server \
    --model models/GPT-OSS-20B-Q6_K.gguf \
    --ctx-size 12288 \
    --port 8502 \
    --threads 8
```

### 2. Create your activity file

Create a new file `activities/my_langchain_chains.py`:

```python
"""Activity 3: Building LangChain chains

Complete the TODO sections to build working chains.
"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize your model (choose one)
llm = ChatOllama(model='qwen2.5:3b', temperature=0.7)

# Or use llama.cpp:
# llm = ChatOpenAI(
#     base_url='http://localhost:8502/v1',
#     api_key='dummy',
#     model='gpt-oss-20b',
#     temperature=0.7
# )
```

### 3. Run and test

```bash
python activities/my_langchain_chains.py
```

## Part 1: Template-based translator

**Goal:** Create a reusable translation chain using prompt templates.

### Your task

Add this code to your file and complete the TODOs:

```python
# Part 1: Translation chain
def build_translator_chain(llm):
    """Build a chain that translates text to any language."""
    
    # TODO: Create a ChatPromptTemplate with:
    #   - System message: "You are a professional translator."
    #   - Human message: "Translate this text to {language}:\n\n{text}"
    prompt = None  # Replace with your prompt template
    
    # TODO: Create a chain: prompt | llm | parser
    chain = None  # Replace with your chain
    
    return chain


# Test it
translator = build_translator_chain(llm)

# Translate to Spanish
result = translator.invoke({
    "language": "Spanish",
    "text": "Hello, how are you today?"
})
print("Spanish:", result)

# Translate to French (reuse same chain!)
result = translator.invoke({
    "language": "French",
    "text": "The weather is beautiful."
})
print("French:", result)
```

### Success criteria

- [ ] Prompt template has two variables: `language` and `text`
- [ ] Chain successfully translates to different languages
- [ ] Same chain works for multiple invocations

### Hints

<details>
<summary>Click to reveal hints</summary>

1. Use `ChatPromptTemplate.from_messages([...])` with tuples
2. Variables are wrapped in curly braces: `{variable_name}`
3. Chain components together with the pipe operator: `|`
4. Use `StrOutputParser()` to get the text output

</details>

## Part 2: Structured data extractor

**Goal:** Extract structured information using Pydantic schemas and JSON parsing.

### Your task

Continue in the same file:

```python
# Part 2: Structured extraction

# TODO: Define a Pydantic model for a book with:
#   - title (str): The book's title
#   - author (str): The author's name
#   - year (int): Publication year
#   - genre (str): Book genre
#   - summary (str): Brief summary
class BookInfo(BaseModel):
    pass  # Replace with your fields


def build_book_extractor(llm):
    """Build a chain that extracts book information."""
    
    # TODO: Create a JsonOutputParser with your BookInfo schema
    parser = None  # Replace with your parser
    
    # TODO: Create a prompt template that:
    #   - Asks the model to extract book information from text
    #   - Includes {format_instructions} in the system message
    #   - Takes {text} as the human message
    prompt = None  # Replace with your prompt
    
    # TODO: Build the chain
    chain = None  # Replace with your chain
    
    return chain, parser


# Test it
extractor, parser = build_book_extractor(llm)

book_text = """
To Kill a Mockingbird by Harper Lee was published in 1960. 
This classic novel of American literature tells the story of 
racial injustice in the Deep South through the eyes of young 
Scout Finch. The book won the Pulitzer Prize and remains a 
staple of high school reading lists.
"""

result = extractor.invoke({
    "text": book_text,
    "format_instructions": parser.get_format_instructions()
})

print("\nExtracted book info:")
print(f"Title: {result['title']}")
print(f"Author: {result['author']}")
print(f"Year: {result['year']}")
print(f"Genre: {result['genre']}")
print(f"Summary: {result['summary']}")
```

### Success criteria

- [ ] BookInfo model has all required fields with descriptions
- [ ] JsonOutputParser uses the BookInfo schema
- [ ] Prompt includes format instructions
- [ ] Chain returns a dictionary with all fields populated

### Hints

<details>
<summary>Click to reveal hints</summary>

1. Pydantic fields use `Field(description="...")` for documentation
2. Create parser: `JsonOutputParser(pydantic_object=YourModel)`
3. Get formatting help: `parser.get_format_instructions()`
4. Pass format instructions to the prompt as a variable
5. The parser automatically converts the model output to a dict

</details>

## Part 3: Multi-step analysis chain

**Goal:** Build a chain that performs multiple operations: summary → sentiment → recommendations.

### Your task

This is the most challenging part! You'll chain multiple operations together.

```python
# Part 3: Multi-step chain

class ReviewAnalysis(BaseModel):
    """Analysis of a product review."""
    overall_rating: int = Field(description="Rating from 1-5 stars")
    sentiment: str = Field(description="positive, negative, or mixed")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    recommendations: str = Field(description="Who should buy this product")


def build_review_analyzer(llm):
    """Build a multi-step chain for review analysis."""
    
    # Step 1: Summarize the review
    # TODO: Create a prompt that asks for a 2-sentence summary
    summary_prompt = None  # Your prompt here
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    # Step 2: Analyze the summary
    # TODO: Create a JsonOutputParser with ReviewAnalysis schema
    parser = None  # Your parser here
    
    # TODO: Create a prompt that:
    #   - Takes {summary} as input
    #   - Analyzes it according to the schema
    #   - Includes {format_instructions}
    analysis_prompt = None  # Your prompt here
    analysis_chain = analysis_prompt | llm | parser
    
    # TODO: Combine both steps
    # Hint: You need to take the output of summary_chain and pass it to analysis_chain
    # One approach: Create a function that calls both chains in sequence
    
    return summary_chain, analysis_chain, parser


# Test it
summary_chain, analysis_chain, parser = build_review_analyzer(llm)

review = """
I bought this laptop three months ago and have mixed feelings. On the positive 
side, the build quality is excellent - it feels premium and sturdy. The battery 
life is impressive, lasting 8-10 hours on a single charge. The screen is bright 
and colors are vibrant.

However, there are some frustrating issues. The keyboard feels cramped and I 
often make typos. The trackpad is overly sensitive and causes accidental clicks. 
Most disappointing is the performance - it struggles with video editing and 
sometimes lags during basic tasks.

For the price point ($1200), I expected better performance. It's good for 
students or office work, but not for creative professionals or gamers.
"""

# Step 1: Summarize
print("Step 1: Summarizing review...")
summary = summary_chain.invoke({"review": review})
print(f"Summary: {summary}\n")

# Step 2: Analyze
print("Step 2: Analyzing summary...")

analysis = analysis_chain.invoke({
    "summary": summary,
    "format_instructions": parser.get_format_instructions()
})

print(f"\nOverall rating: {analysis['overall_rating']}/5 stars")
print(f"Sentiment: {analysis['sentiment']}")
print(f"\nPros:")

for pro in analysis['pros']:
    print(f"  + {pro}")

print(f"\nCons:")

for con in analysis['cons']:
    print(f"  - {con}")

print(f"\nRecommendations: {analysis['recommendations']}")
```

### Success criteria

- [ ] Summary chain produces a concise 2-sentence summary
- [ ] Analysis chain extracts all structured fields
- [ ] Both chains work together in sequence
- [ ] Output includes rating, sentiment, pros, cons, and recommendations

### Hints

<details>
<summary>Click to reveal hints</summary>

1. The summary prompt should take `{review}` and ask for brevity
2. The analysis prompt takes `{summary}` (not the original review)
3. Don't forget to include `{format_instructions}` in your analysis prompt
4. To chain them: first invoke summary_chain, then pass result to analysis_chain
5. Advanced: You can create a single chain by defining a custom function that combines both

</details>


## Next steps

- Explore [LangChain documentation](https://python.langchain.com/)
- Try the **LangChain Expression Language (LCEL)** for more advanced chaining
- Build a Gradio UI around your chains (see Demo 5)
- Experiment with **streaming** outputs using `.stream()`
- Learn about **LangGraph** for more complex agentic workflows
