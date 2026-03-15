from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage, HumanMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
config = {"configurable": {"thread_id": "1"}}

class CustomState(AgentState):
    favoriate_color: str

# ========================================================== #
# ===================== Write to State ===================== #
# ========================================================== #
@tool
def update_favoriate_color(favoriate_color: str, runtime: ToolRuntime) -> Command:
    """Update the favoriate color of the user in the state once they've revealed it"""
    return Command[tuple[()]](
        update={
            "favoriate_color": favoriate_color,
            "messages": [ToolMessage(
                content="Successfully updated favoriate color.",
                tool_call_id=runtime.tool_call_id
            )]
        }
    )

agent = create_agent(
    model=model,
    tools=[update_favoriate_color],
    checkpointer=InMemorySaver(),
    state_schema=CustomState
)

# ========================================================== #
# ================= Update State: Option 1 ================= #
# ========================================================== #

response = agent.invoke(
    {
        "messages": [HumanMessage(content="My favoriate color is green.")]
    },
    config=config
)

# ========================================================== #
# ================= Update State: Option 2 ================= #
# ========================================================== #
response = agent.invoke(
    {
        "messages": [HumanMessage(content="My favoriate color is yellow.")],
        "favoriate_color": "yellow"
    },
    config=config
)

pprint(response)

# ========================================================== #
# ====================== Read State ======================== #
# ========================================================== #
@tool
def read_favoriate_color(runtime: ToolRuntime) -> str:
    """Read the favoriate color of the user from the state"""
    try:
        return runtime.state["favoriate_color"]
    except KeyError:
        return "No favoriate color found in state"
    
agent = create_agent(
    model=model,
    tools=[update_favoriate_color, read_favoriate_color],
    checkpointer=InMemorySaver(),
    state_schema=CustomState
)

response = agent.invoke(
    {
        "messages": [HumanMessage(content="My favoriate color is blue.")]
    },
    config=config
)
pprint(response)

response = agent.invoke(
    {
        "messages": [HumanMessage(content="What's my favoriate color?")]
    },
    config=config
)
pprint(response)