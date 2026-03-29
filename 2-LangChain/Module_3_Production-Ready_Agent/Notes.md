## Production-Ready Agent
### 1. Middleware
It is a catch-all term used for functions that can be **inserted within the agent loops**, which allows us to control and customize the agent's execution.   
For example, in a refund system, we can insert a human-approval middleware fuction between the model and the tool to add a human oversight.  
It is the key to leveling up a hobby project to aan agent ready for real users.
![Middleware](../../resources/Middleware.png)

### 2. Node-style Middleware
#### 2.1 Managing Long Conversations
As conversation grow, it hits the LLM's **context window limit**.
- Context window = max input size per single LLM call
- LLM has zero memory itself, it is stateless - **the app re-sends everything each time and it wakes up fresh**. (system prompt + tool description + history + current message)
- InMemorySaver is just a database holding messages so the app can re-send them next call.

Middleware compresses/Trims the message list before each LLM call to avoid context window overflow within a session.  
Moving from maintaining **short conversations with checkpointer** to **long conversations using middleware** in two ways:
- Summarizing the conversation.
- Trimming or deleting the messages.
    - @before_agent / @ after_agent: run **once** per run
    - @before_model / @ after_model: run **multiple times** per run  

The **checkpointer** is like a **database** - it saves and loads **the full unmodified history**.  --> Solves persistence.
Middleware only affects what the LLM sees in that moment - it doesn't permanently delete from the checkpointer's storage, unless using RemoveMessage which does permanently delete. --> Solves overflow.

#### 2.2 Human In The Loop
- Approving sensitive actions
    - Approve
    - Reject
    - Edit: Edit then approve the editted version immediately.
        ```python 
        from langchain.agents.middleware import HumanInTheLoopMiddleware
        agent = create_agent(
            model=model,
            tools=[read_email, send_email],
            checkpointer=InMemorySaver(),
            state_schema=EmailState,
            middleware=[
                HumanInTheLoopMiddleware[AgentState, None](
                    interrupt_on={
                        "read_email": False, # Choose which tools require human in the loop approval
                        "send_email": True
                    },
                    description_prefix="Send email tool execution requires approval."
                )
            ],
            system_prompt="You are an email assistant. Always delegate tasks to read_emial then send_email." \
            "You are not allowed to ask followup questions when invoke read_email tool."
        )
        ```
- Adding missing context
- Debugging agents

### 3. Wrap Style Middleware (Dynamic Agents)
Unlike the above **node-style middleware** like trimming messages, which is inserted into agent's runtime by hooking it before or after either the model or agent was invoked.   
This **wrap style middleware** is to hook into the model instance itself. It isn't before or after the model, this is the model, for instance, change prompts, tools and even the model.
- **Model Request**: The model instance is represented as a model request, it is the object the middleware receives before model execution, including:
    - System prompt, 
    - Available tool calls, 
    - The state and the foundational model itself  
So if you can grab the model request and adjust that with your function, you're essentially adjusting what the model looks like. 
![Model Request](../../resources/Model_Request.png)
#### 3.1 Dynamic Prompts
- @dynamic_prompt:  
Use ```@dynamic_prompt``` to tell the agent instead of a fixed system prompt string, runs this function before each LLM call to generate the prompt dynamically. Without this ```@dynamic_prompt```, system prompt is fixed forever at agent creation.  
Without this decorator, even the context is passed and stored, nothing reads it, the system prompt still remains unchanged.
- request:ModelRequest   
In a ```@dynamic_prompt``` function, use ```request.runtime.context.xxx``` as it is inside a middleware function that runs before the LLM call. While if inside a ```@tool```, use ```runtime.context.xxx``` directly.
    ```python 
    from langchain.agents.middleware import dynamic_prompt, ModelRequest
    @dataclass
    class LanguageContext:
        user_language: str = "English"
    @dynamic_prompt
    def user_language_prompt(request: ModelRequest) -> str:
        """Generate system prompt based on user role."""
        user_language = request.runtime.context.user_language
    ```

#### 3.2 Dynamic Tool Calls
- @wrap_tool_calls  
Wraps each LLM call, you control request and response of the model, you can adjust the tool calls that the model sees, for example, you can add a tool call to the list of available tools, or remove some tools from the list.
    - Use ```request.override()``` to override the model request with a new one rather than directly modifying request because ModelRequest is immutable — it's a clean functional pattern where you create a modified copy rather than changing the original.
        ```python
        # ❌ wrong — ModelRequest is immutable
        request.tools = [web_search]
        # ✅ correct — creates a new request with tools swapped
        request = request.override(tools=[web_search])
        ```
    - Handler: 
        - Unlike @dynamic_prompt, which is only a prompt generator, its only job is to return a strng. **The middleware pipeline takes that string and handles everything else internally.**
        - @wrap_model_call gives you full control of the entire LLM call, which means **you are responsible for actually triggering it**. You must call this handler to actually execute LLM call, otherwise the model will never be called and the agent will never move forward. 
            ```python
            from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
            from typing import Callable

            @wrap_model_call
            def dynamic_tool_call(
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse]
            ) -> ModelResponse:
                """Dynamically call tools based on the runtime context"""
                user_role = request.runtime.context.user_role
                if user_role == "internal":
                    pass # internal users get access to all tools
                else:
                    tools = [web_search]
                    request = request.override(tools=tools)
                return handler(request) # must call LLM here, you decide when to call it and with what request
            ```


