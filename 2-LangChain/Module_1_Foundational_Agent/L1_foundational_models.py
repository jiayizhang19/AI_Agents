from dotenv import load_dotenv
from pprint import pprint
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

load_dotenv()

# ================================================================ #
# ================= Initialize a Model/Chatbot =================== #
# ================================================================ #
gemini_model = "gemini-3-flash-preview"
model = ChatGoogleGenerativeAI(model=gemini_model)
response = model.invoke("What's the captical of the Moon?")
print(response.text)

# ================================================================ #
# =================== Initialize an Agent ======================== #
# ================================================================ #
# agent_model = "gemini-3.1-pro-preview" # This will direct to vertexai instead of gemini model
agent = create_agent(model=model)
# ====== Interact with the agent with questions using the "invoke" ==== #
response = agent.invoke(
    {"messages":[HumanMessage(content="What's the captical of the Moon?")]}
)
# pprint(response)
pprint(response["messages"][-1].content[0]["text"])

response_to_ai_message = agent.invoke(
    {"messages": [
        HumanMessage(content="What's the captical of the Moon?"),
        AIMessage(content="The captical of the Moon is Luna City"),
        HumanMessage(content="Interesting, tell me more about Luna City")
    ]}
)
# pprint(response_to_ai_message)
pprint(response_to_ai_message["messages"][-1].content[0]["text"])


# ================================================================ #
# ============= Stream Tokens to Conduce Latency ================= #
# ================================================================ #
for token, metadata in agent.stream(
    {"messages": [HumanMessage(content="Tell me about Luna City, the capital of the Moon.")]},
    stream_mode="messages"
):
    if token.content:
        print(token.content[0]["text"], end="", flush=True)

# ================================================================ #
# =================== Create System Prompts ====================== #
# ================================================================ #
system_prompt = "You are a science fiction writer, create a capital city at the users request."
scifi_agent = create_agent(
    model=model,
    system_prompt=system_prompt
)
# Enable streaming by using agent.stream(). If instead using agent.invoke(), you will get the full answer at once.
# flush=True only makes the streamed output visible immediately and smoothly in the terminal, instead of showing chunks.
for token, metadata in scifi_agent.stream(
    {"messages": [HumanMessage(content="What's the captical of the Moon?")]},
    stream_mode="messages"
):
    if token.content:
        # flush=True forces python to immediately write to the screen rather than buffering output before displaying.
        print(token.content[0]["text"], end="", flush=True)

# ================================================================ #
# ================== Create structured output ==================== #
# ================================================================ #
class CapticalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

scifi_stru_agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    response_format=CapticalInfo
)
question = HumanMessage(content="What's the capital of the Moon?")
response = scifi_stru_agent.invoke(
    {"messages":[question]}
)
# pprint(response)
response["structured_response"]
capital_name = response["structured_response"].name
location = response["structured_response"].location
print(f"{capital_name} is a city located at {location}.")