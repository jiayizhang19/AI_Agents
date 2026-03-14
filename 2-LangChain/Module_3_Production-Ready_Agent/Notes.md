## Production-Ready Agent
### Middleware
It is a catch-all term used for functions that can be **inserted within the agent loops**, which allows us to control and customize the agent's execution.   
For example, in a refund system, we can insert a human-approval middleware fuction between the model and the tool to add a human oversight.  
It is the key to leveling up a hobby project to aan agent ready for real users.
![Middleware](../../resources/Middleware.png)

### Managing Long Conversations
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

The **checkpointer** is like a **database** - it saves and loads **the full unmodified history**.  -- Solves persistence.
Middleware only affects what the LLM sees in that moment - it doesn't permanently delete from the checkpointer's storage, unless using RemoveMessage which does permanently delete. --> Solves overflow.