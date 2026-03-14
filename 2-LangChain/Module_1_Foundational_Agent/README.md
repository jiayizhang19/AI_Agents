# Module 1: Foundational Agent

A hands-on LangChain tutorial module covering the building blocks of AI agents.

## Overview

This module introduces foundational concepts for building AI agents using LangChain. Each lesson file builds upon the previous, progressing from basic model interactions to full-featured agents with tools and memory.

## Project Structure

```
Module_1_Foundational_Agent/
├── L1_foundational_models.py    # Core model & agent concepts
├── L2.1_tools_calculation.py    # Tool definition & usage
├── L2.2_tools_web_search.py     # Web search tool integration
├── L3_short-term_memory.py     # Conversation memory
├── L4_multimodal_messages.py   # Image handling
└── Notes.md                    # Learning notes
```

## Lessons

### L1: Foundational Models
- Initializing a chat model with `ChatGoogleGenerativeAI`
- Creating agents with `create_agent()`
- Streaming tokens to reduce perceived latency
- Customizing behavior with system prompts
- Structured output using Pydantic models (`BaseModel`)

### L2.1: Tools - Calculation
- Defining tools with the `@tool` decorator
- Customizing tool names and descriptions
- Invoking tools standalone vs. within an agent
- The `invoke()` pattern for LangChain tools

### L2.2: Tools - Web Search
- Integrating external APIs (Tavily) as tools
- Real-time information retrieval
- Tool invocation for live data

### L3: Short-Term Memory
- Maintaining conversation context across turns
- Using `InMemorySaver` checkpointer from LangGraph
- Thread IDs for grouping conversation states
- How agents retain and recall information

### L4: Multimodal Messages
- Encoding images to Base64
- Sending multimodal messages (text + images)
- Model-specific message formats (Gemini vs OpenAI)

## Key Concepts

| Concept | Description |
|---------|-------------|
| `invoke()` | Universal execution method in LangChain for models, agents, chains, and tools |
| `stream()` | Streams tokens incrementally to reduce perceived latency |
| System Prompt | Customizes agent personality and output structure |
| Structured Output | Forces model responses into defined Pydantic schemas |
| `@tool` decorator | Converts Python functions into LangChain Tool objects |
| Checkpointer | Saves conversation state snapshots for memory persistence |
| Thread ID | Groups related checkpointer states together |

## Dependencies

```
langchain
langchain-google-genai
langgraph
python-dotenv
pydantic
tavily-python
```

## Environment Setup

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Running the Lessons

```bash
python L1_foundational_models.py
python L2.1_tools_calculation.py
python L2.2_tools_web_search.py
python L3_short-term_memory.py
python L4_multimodal_messages.py
```

## Model Used

Primary model: `gemini-3-flash-preview` (Google Generative AI)

Alternative models can be configured by changing the model string in each file.

## Notes on Latency

Agent systems have response times measured in seconds, unlike traditional software with millisecond responses. **Streaming tokens** helps reduce perceived latency by showing output as it's generated rather than all at once.