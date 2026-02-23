## Workflows V.S. Agents
An agentic framework is not always needed when building an application around LLMs. Sometimes, predefined **workflows** are sufficient to fulfill user requests. However, when the workflow becomes more complex, such as letting an LLM call functions or using multiple agents, these abstractions start to become helpful.
- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
- **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks. (Choose the right framework for your needs.)

## Agentic Framework - Smolagents
- A simple yet powerful framework for building AI agents.
  (Check more resources at the end of the blog https://huggingface.co/learn/agents-course/unit2/smolagents/introduction)
- Primary Advantages: It supports a code-first approach with minimal abstractions, letting agents interact directly via Python function calls.
### Tools
- Pre-built tools: 
  - DuckDuckGoSearchTool: search on the web.
  - GoogleSearchTool
  - VisitWebpageTool
  - ....
- Use automatic code generation with **additional_authorized_imports** without explicitly specified any tools (available for CodeAgent while not for ToolCallingAgent), when:
  - The task is purely computational;
  - You trust the LLM to write correct logic;
  - No external systems are involved;
  - Failure has low consequences.
- Custom tools:
  - When to use:
    - You need to interact with external systems, for example, query DB or send email, etc..;
    - Business logic is complex / important;
    - Security / safety is a concern.
  - Methods of creating tools:
    - Use decorator @tool for simple function-based tools
      ```python
      from smolagents import tool
      @tool
      def specific_tool():
        pass
      ```
    - Use subclasses of Tool to add more flexibility for complex functionality or custom metadata. 
      ```python
      from smolagents import Tool
      class SpecificTool(Tool):
        ...
      ```
- Importing a Tool from the Hub, Hugging Face Space, LangChain, MCP server
  - Hub:
    ```python
    from smolagents import load_tool
    xxx_tool = load_tool()
    ```
  - Hugging Face Space:
    ```python
    from smolagents import Tool
    xxx_tool = Tool.from_space()
    ```
  - LangChain:
    ```python
    pip install -U langchain-community
    from smolagents import Tool
    xxx_tool = Tool.from_langchain()
    ```
  - MCP server:
    ```python
    pip install "smolagents[mcp]"
    from mcp import StdioServerParameters
    server_parameters = StdioServerParameters(
      command="uvx",
      args=["--quiet", "pubmedmcp@0.1.3"],
      env={"UV_PYTHON": "3.12", **os.environ},
    )
    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
      agent = CodeAgent(tools=[*tool_collection.tools], model=model, add_base_tools=True)
      agent.run("Please find a remedy for hangover.")
    ```
### LLM Intergration 
- InferenceClientModel(): the HF inference providers router, burns credits.
  ```python
  model=InferenceClientModel()
  ```
- **LiteLLMModel**: Leverages LiteLLM for lightweight model interactions, one interface for 100+ providers (OpenAI, Anthropic, Gemini, Groq, etc.)
  - Bypass HF billing with gemini model. (Setting up a Gemini API Key via Google AI Stuido https://aistudio.google.com/ and check out available model https://ai.google.dev/gemini-api/docs)
    ```python
    gemini_model = LiteLLMModel(
      model_id="gemini/gemini-2.5-flash",
      api_key=os.getenv("GEMINI_API_KEY")
    )
    ```
- TransformersModel: Implements a **local** transformers pipeline for seamless integration. (local download + local execution).
- OpenAIServerModel: Connects to any service that offers an OpenAI API interface, unlike LiteLLMModel.
- AzureOpenAIServerModel: Supports integration with any Azure OpenAI deployment.
### Agent Types
- **Code Agents**: Code generation paradigm and the default agent type in smolagents. It writes actions in code rather than JSON. However, research shows that **tool-calling LLMs work more effectively with code directly.**
  - Example of agent's trace:
    ```python
    Executing parsed code:
    music_recommendations = web_search(query="best music recommendations for a party")       
    print("Music recommendations:"music_recommendations)   
    ```
  - Running a Code Agent inside smolagents:
    ```python
    web_and_tools_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(),suggest_menu],
    model=gemini_model,
    # model=InferenceClientModel(), # It needs credits
    )
    web_and_tools_agent.run(
        "Search for the best music recommendations for a party."
        "Prepare me a menu for a party that is informal."
    )
    ```
- **Tool Calling Agents**: Structured tool-calling paradigm. These agents use the built-in tool-calling capabilities of LLM providers to generate tool calls as JSON structures. This is the standard approach used by OpenAI, Anthropic, and many other providers.
  - Example of agent's trace, instead of seeing Executing parsed code, you'll see something like:
    ```
    Calling tool: 'web_search' with arguments: {'query': 'best music recommendations for a party'}   
    ```
  - Running a Tool Calling Agent inside smolagents
    ```python
    from smolagents import ToolCallingAgent, WebSearchTool, LiteLLMModel

    tool_calling_agent = ToolCallingAgent(
      tools=[DuckDuckGoSearchTool(),suggest_menu],
      model=gemini_model,
    )

    tool_calling_agent.run(
      "Search for the best music recommendations for a party."
      "Prepare me a menu for a party that is informal."
    )
    ```
  
## Multi-Agent System
Multi-agent systems enable **specialised agents to collaborate on complex tasks**, improving **modularity, scalability, and robustness**. Instead of relying on a single agent, tasks are distributed among agents with distinct capabilities.  
A typical setup might include:
  - A **Manager Agent** for task delegation
  - A **Code Interpreter Agent** for code execution
  - A **Web Search Agent** for information retrieval

See example on HF Agents Course - Multi-Agent Systems.