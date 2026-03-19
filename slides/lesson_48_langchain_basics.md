---
marp: true
theme: default
paginate: true
style: |
  section {
    background-color: #1a1a2e;
    color: #e0e0e0;
  }
  section h1, section h2, section h3 {
    color: #e2b55a;
  }
  section a {
    color: #93c5fd;
  }
  section strong {
    color: #f0f0f0;
  }
  section table th {
    background-color: #2a2a4a;
    color: #e2b55a;
  }
  section table td {
    background-color: #1e1e36;
    color: #e0e0e0;
  }
  section code {
    background-color: #2a2a4a;
    color: #e0e0e0;
  }
  section pre {
    background-color: #12122a;
  }
  section::after {
    color: #888;
  }
---

# Lesson 48: LangChain basics

**Building structured LLM applications**

---

## Recap: where we are

So far you've learned:
- **Lesson 44:** State of the art in generative AI
- **Lesson 45:** LLM deployment strategies
- **Lesson 46:** Prompting fundamentals
- **Lesson 47:** Advanced prompting techniques (ReAct agents)

**Today:** How to use **LangChain** to structure and scale your LLM applications

---

## What is LangChain?

LangChain is a **framework for developing applications powered by language models**.

**Key philosophy:** Applications are more powerful when they are:
- **Context-aware:** Connect LLMs to sources of context (documents, APIs, databases)
- **Reasoning-enabled:** Rely on LLMs to reason about context and take actions

**Think of it as:** A toolkit for building complex LLM workflows without writing everything from scratch

---

## Today's outline

We'll explore four core LangChain concepts:

1. **Chat models and LLM wrappers** - Unified interface to different model providers
2. **Chat prompt templates** - Structured, reusable prompts
3. **Output parsers** - Extracting structured data from LLM responses
4. **Basic chains** - Composing multiple steps into workflows

After the slides: **Demo 5** will show these concepts in action, then **Activity 3** for hands-on practice.

---

# Chat models and LLM wrappers

Unified interfaces to different model providers

---

## Chat models and structured messages

LangChain provides a **unified interface** to different model providers (Ollama, OpenAI, Anthropic, etc.).

**Same interface, different providers.** Easy to swap!

**Chat models use structured messages:**
- `SystemMessage` - Instructions for the model
- `HumanMessage` - User input
- `AIMessage` - Model response
- `ToolMessage` - Tool execution results (for agents)

---

# Chat prompt templates

Building structured, reusable prompts

---

## The problem: string formatting is fragile

**Manual approach:** Manual string formatting is easy to get wrong:
- Easy to make mistakes with string formatting
- What if the model expects a different format?
- What if you want to add few-shot examples?
- What if you need to reuse this across multiple places?

**Problem:** Hard to maintain, error-prone, not reusable

---

## ChatPromptTemplate: structured prompts

**ChatPromptTemplate** provides a structured way to build prompts with variables.

**Benefits:**
- Type-safe (validates required variables)
- Reusable across your application
- Supports multiple message types
- Easy to test and version control

---

## Template variables and formatting

Templates support **f-string style placeholders** in curly braces: `{variable_name}`

**Multiple inputs** can be passed through `.invoke()` as a dictionary.

**Missing variables raise an error** - helps catch bugs early!

---

## Few-shot prompting with templates

Include examples to guide the model by alternating human and AI messages in the template.

**Pattern:** Alternate `("human", ...)` and `("ai", ...)` for examples

**Example:** A sentiment classifier can include examples like "I love this product!" → "positive", "This is terrible." → "negative", etc.

---

## MessagesPlaceholder: dynamic messages

Sometimes you need to insert a **variable number of messages** (e.g., conversation history).

**MessagesPlaceholder** allows you to pass in conversation history as a list of messages.

Useful for chatbots that maintain context!

---

# Output parsers

Extracting structured data from text responses

---

## The problem: LLMs return strings

Models generate text, but you often need **structured data**.

**Challenges:**
- How do you extract lists reliably?
- What if the format changes slightly?
- What if you need JSON, not text?

**Solution:** Output parsers guide the model to **produce structured output** and **parse it automatically**.

---

## StrOutputParser: simple string extraction

The simplest parser just extracts the content string from the AIMessage object.

**Use when:** You just need plain text, no structure needed

---

## JsonOutputParser: structured data

**JsonOutputParser** extracts JSON objects from model responses using Pydantic schemas.

**Define a Pydantic model** with field descriptions, then create a parser from it.

**The parser automatically generates formatting instructions** that you can inject into your prompt!

---

## JsonOutputParser example

**Input:** "Sarah Chen is a 34-year-old software engineer."

**Output (as Python dict):**
- `name`: "Sarah Chen"
- `age`: 34
- `occupation`: "software engineer"

**Benefits:**
- Validates structure against Pydantic schema
- Automatically retries if JSON is malformed
- Type-safe outputs

---

## PydanticOutputParser: even stricter

**PydanticOutputParser** handles complex nested structures with custom validation.

**Can include Pydantic validators** to enforce business rules (e.g., "time cannot be negative").

Returns **Pydantic objects** with full validation - not just dicts!

---

## CommaSeparatedListOutputParser

For simple lists, **CommaSeparatedListOutputParser** automatically splits on commas and strips whitespace.

**Example output:** `['Python', 'JavaScript', 'Java', 'C++', 'Go']`

---

## Output parser format instructions

All parsers provide **format instructions** via `.get_format_instructions()` that you can inject into prompts.

**Example output:**
"The output should be formatted as a JSON instance that conforms to the JSON schema below..."

This **guides the model** to produce the right format!

---

# Basic chains

Composing steps into workflows

---

## What is a chain?

A **chain** connects multiple components into a workflow:

**prompt_template → model → output_parser → result**

**The LCEL (LangChain Expression Language) syntax:** `chain = prompt | llm | output_parser`

The `|` operator (pipe) connects components, similar to Unix pipes.

---

## Simple chain example

**Define components:**
1. Prompt template with system and human messages
2. Chat model (e.g., ChatOllama)
3. Output parser (e.g., StrOutputParser)

**Compose into a chain:** `chain = prompt | llm | parser`

**Use the chain:** `result = chain.invoke({"question": "What is Python?"})`

**Each `|` passes the output of one component as input to the next.**

---

## How chains work

When you call `chain.invoke(input)`:

1. **Prompt template** receives the input dict, produces messagesLangChain automatically uses the messages to construct the prompt
2. **LLM** receives messages, generates a response (AIMessage)
3. **Output parser** receives AIMessage, extracts/parses content
4. **Final result** is returned to you

**Type system ensures compatibility** - LangChain validates that outputs match expected inputs at each step.

---

## Chain with structured output

Combining templates, models, and parsers for structured extraction:

**Example:** Todo item extractor
- **Input:** "I need to finish the report by Friday, it's high priority"
- **Output:** `{"task": "finish the report", "priority": "high", "due_date": "2026-03-21"}`

**Pattern:** Define Pydantic model → Create JsonOutputParser → Build prompt with format instructions → Chain together

---

## Reusable chains

Chains are **objects you can reuse**:

**Define once, use many times** - create a chain and invoke it with different inputs.

**Example:** A summarizer chain can be reused with different documents and word limits.

**Benefits:**
- Define logic once, reuse everywhere
- Easy to test and version control
- Can be composed into larger chains

---

## Sequential chains

Chain outputs can feed into subsequent chains:

**Example pattern:**
1. First chain extracts entities from text
2. Second chain uses those entities to create a summary
3. Manually compose by calling chains in sequence

More complex compositions use **RunnableSequence** (advanced topic).

---

## Chain benefits recap

**Why use chains instead of manual code?**

1. **Composability** - Build complex workflows from simple pieces
2. **Reusability** - Define once, use many times
3. **Type safety** - Automatic validation of inputs/outputs
4. **Streaming support** - Built-in streaming for all components
5. **Monitoring** - LangSmith integration for observability
6. **Testing** - Easy to unit test individual components

**In production:** Chains make LLM applications more maintainable and debuggable.

---

## What we covered today

1. **Chat models** - Unified interface to different LLM providers
2. **Chat prompt templates** - Structured, reusable prompts with variables
3. **Output parsers** - Extract structured data from text responses
4. **Basic chains** - Compose components into workflows with LCEL

**Next steps:**
1. **Demo 5:** See these concepts in a working application
2. **Activity 3:** Build your own LangChain chains

---

## Additional resources

**LangChain documentation:**
- [Chat models](https://python.langchain.com/docs/modules/model_io/chat/)
- [Prompt templates](https://python.langchain.com/docs/modules/model_io/prompts/)
- [Output parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [LCEL (chains)](https://python.langchain.com/docs/expression_language/)

**LangSmith:** Debugging and monitoring platform for LangChain apps
- [langsmith.com](https://www.langsmith.com/)

---

# Questions?

**Next:** Demo 5 - LangChain in action

Then: Activity 3 - Build your own chains
