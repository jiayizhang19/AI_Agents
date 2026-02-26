## Foundational Models
### Terminologies
- temperature: control the randomness of the model's output.
a higher number makes the responses more creative, a lower one makes them more deterministic.
- max token: limits the total number of tokens in the response, controlling how long the output can be.
- timeout: the max time to wait for a response from the model before cancelling their request
- max retries: the max amount of times to retry a request if that request fails.
### Universal Method
- .invoke() = LangChain's universal execution method as LangChain tools use **JSON-style arguments**. It's widely used in model, agent, chain and tool.
    |.invoke()|normal python function call|
    |--|--|
    |square_root.invoke({"x": 121})|square_root(x=121)|
    |Tool wrapper, converts the function into a Tool object|Direct execution|
### Issues with Agent Systems - Latency
The agent will be running with response times of seconds, even minutes, unlike software systems with response times in milliseconds.
- One way to conduce the perceived latency of the system is by **streaming tokens** to the user as they appear, rather than printing the answer all at once.
### Customize the Performance of a Chat Model - System Prompts
You can provide any description of the agent and output structure in the system prompt.
### Tracing Your Agent - LangSmith
A tool to visualize:
- Results returned from the agent
- Latency
- Token usage
- ...
### Short-Term Memory - Maintain memory over the length of the conversation
LangChain is tracking messages in something called the state - the memeory of the agent.
- checkpointer: Save a snapshot of the state at the end of each run, and then groups.
### Multimodal Messages
Encode image and audio files in Base 64
