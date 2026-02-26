import base64
from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# The file is relative to the current working diretory (CWD), not relative to where the script is located.
with open("resources/NVIDIA.png", "rb") as f:
    img_bytes = f.read()
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

load_dotenv()
gemini_model = "gemini-2.5-flash"

# For Gemini model
multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Tell me about the image"},
    {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
])

question = HumanMessage(content="How many human are there in this image?")

# For OpenAI model
# multimodal_question = HumanMessage(content=[
#     {"type": "text", "text": "Tell me about the image"},
#     {"type": "image", "image": img_b64, "mime_type": "image/png"}
# ])
config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model = ChatGoogleGenerativeAI(model=gemini_model),
    checkpointer=InMemorySaver(),
)

response = agent.invoke(
    {"messages": [multimodal_question]},
    config=config
)

pprint(response["messages"][-1].content)

response = agent.invoke(
    {"messages":[question]},
    config=config
)
pprint(response["messages"][-1].content)