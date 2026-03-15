from dataclasses import dataclass
from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime

# Pass useful context to the agent when it runs.

# ================================================================= #
# ===================== Create Context Schema ===================== #
# ================================================================= #
@dataclass
class ColorContext:
    fav_color: str = "blue"
    least_fav_color: str = "black"

# ================================================================= #
# ============== Make Tool Calls to Access Context ================ #
# ================================================================= #
@tool
def get_fav_color(runtime: ToolRuntime) -> str:
    """Get the favouriate color of the user"""
    return runtime.context.fav_color

@tool
def get_least_fav_color(runtime: ToolRuntime) -> str:
    """Get the least favourite color of the user"""
    return runtime.context.least_fav_color

# ================================================================= #
# ==================== Pass Context to the Agent ================== #
# ================================================================= #
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Define the structure at agent creation time using context_schema=ColorContext.
# Otherwise the ToolRuntime.context will always be None as the injection system is inactive.
agent = create_agent(
    model=model,
    tools=[get_fav_color, get_least_fav_color],
    context_schema=ColorContext
)
# Pass the data at execution time using context=ColorContext()
response = agent.invoke(
    {"messages": [HumanMessage(content="What's my faviorate color?")]},
    context=ColorContext()
)
pprint(response["messages"][-1].content)

# ================================================================= #
# =============== Change the Context Default Value ================ #
# ================================================================= #
response = agent.invoke(
    {"messages": [HumanMessage(content="Which color do i like the most?")]},
    context=ColorContext(fav_color="green")
)
pprint(response["messages"][-1].content)