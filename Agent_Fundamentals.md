### I. Definition of the Agent
The agent is an AI model capable of **reasonig, planning, and interacting with its environment (acting)**. Think of the Agent as having two main parts:
1. **The Brain (AI model)**: Handles **reasoning and planning**, it decides which actions to take based on the situation.
    - LLMs are the most common type of AI models for this purpose.
    - LLMs are a key component of AI Agents, providing the foundation for **understanding and generating human language**. 
    - The input sequence you provide an LLM is called a prompt. Careful design of the prompt makes it easier to guide the generation of the LLM toward the desired output.
    - LLM internal knowledge only includes events prior to their training.
2. **The Body (Capabilities and Tools)**: Represents everything the Agent is equipped to do. The scope of possible **actions** depends on what the agent has been equipped with.  

An agent can perform any task we implement via **Tools** to complete **Actions**. (Actions are not the same as Tools. An action, for instance, can involve the use of multiple Tools to complete.)

### II. Messages: The Underlying System of LLMs
1. **System messages** (System prompts): Defined how the model should behave. They serve as **persistent instructions**, guide every subsquent interaction.
   - It also gives information about the **available tools**, provides instructions to the model on how to format the actions to take, and includes guidelines on how the thought process should be segmented.
2. Conversations: It consists of User and Assistant(LLM) messages.
   - Chat templates help maintain context by preserving conversation history, stroing previuos exchanges between the user and the assistant. This leads to more coherent **multi-turn conversations**.
  
### III. Tools
LLMs can only receive text inputs and generate text outputs. They have no way to call tools to on their own. When we talk about providing tools to an Agent, we mean teaching the LLM about **the existence of these tools** and instructing it to **generate text-based invocations** when needed.
1. Definition:
A tool is **a function** given to the LLM, which should fulfill **a clear objective**. It should contain:
    - A **textual description** of what the function does.
    - A callable function to perform an action.
    - **Input typings**.
    - **Output typings**.

Tip: Use @tool decorator for auto-formatting above tools_description in system messages. An example of system message:
```
system_message = """ You are an AI assistant designed to help users efficiently and accurately. Your primary goal is to provide helpful, precise, and clear responses.

You have access to the following tools:
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
"""
```
2. MCP (Model Context Protocol): a unified tool interface
MCP is an open protocol that standardizes **how applications provide tools to LLMs**. It provides:
    - A growing list of pre-built integrations that your LLM can directly plug into.
    - The flexibility to switch between LLM providers and vendors.
    - Best practices fro securing your data within your infrastructure.

This means that any framework implementing MCP can **leverage tools defined within the protocol**, eliminating the need to reimplement the same tool interface for each framework.


### IV. Agent Workflow
Agents' work is a continuous cycle of **Thought --> Action --> Observation**. In many Agent frameworks, the rules and guidelines are embedded directly into the **system prompt**, ensuring that every cycle adheres to a defined logic.
#### Thought: Internal Reasoning and the ReAct(Reasoning + Acting) Approach
- Chain-of-Thought(CoT): 
    - A prompt technique that guides a model to **think through a problem step by step** before producing a final answer.
    - This approach helps the model **reason internally**, especially for logical or mathematical tasks, **without interacting with external tools**.
- ReAct (Reasoning + Acting):
    - A prompt technique encourages the model to think step-by-step and **interleave actions** (like **using external tools**) between **reasoning steps**, which **combines "Reasoning"** (Think) with **"Acting"** (Act).
#### Action: Tool Usage
- Types of Agent:
    - JSON Agent: The action to take is specified in JSON format.
    - Code Agent: The action writes a code block that is interpreted externally.
    - Function-calling Agent: A subcategory of the JSON Agent which has been fine-tuned to generate a new message for each action.
- The Stop and Parse Approach:
    - Generation in a structued format: The agent outputs its intended action in a clear, predetermined format (JSON or code).
    - Halting further generation: Once the text defining the action has been emitted, **the LLM stops generating additional tokens**.
    - Parsing the output: An external parser reads the formatted action, determines which Tool to call, and extracts the required parameters.
#### Observation: Integrating Feedback to Reflect and Adapt
Observations are **how an Agent perceives the consequences of its actions.**
Each cycle allows the agent to incorporate fresh information (observations) into its reasoning (thought), ensuring that the final answer is well-informed and accurate, which is called **dynamic adaption**. It contains:
    - **Collects Feedback**: Receives data or confirmation that its action was successful (or not).
    - **Appends Results**: Integrates the new information into its existing context, effectively updating its memory.
    - **Adapts its Strategy**: Uses this updated context to refine subsequent thoughts and actions.
