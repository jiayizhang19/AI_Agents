from dataclasses import dataclass
from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime


@dataclass
class ColorContext:
    fav_color: str = "blue"
    least_fav_color: str = "black"

@tool
def get_fav_color(runtime: ToolRuntime) -> str:
    """Get the favouriate color of the user"""
    return runtime.context.fav_color

@tool
def get_least_fav_color(runtime: ToolRuntime) -> str:
    """Get the least favourite color of the user"""
    return runtime.context.least_fav_color

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

agent = create_agent(
    model=model,
    tools=[get_fav_color, get_least_fav_color],
    context_schema=ColorContext
)

response = agent.invoke(
    {"messages": [HumanMessage(content="What's my faviorate color?")]},
    context=ColorContext()
)

pprint(response["messages"][-1].content)