from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command


load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email"""
    # take email from state
    return runtime.state["email"]

@tool
def send_email(runtime: ToolRuntime) -> str:
    """Send email."""
    # fake email sending
    return f"Email sent."

class EmailState(AgentState):
    email: str

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

config = {"configurable":{"thread_id": "email"}}
response = agent.invoke(
    {
        "messages": [HumanMessage(content="Please read my email and send a response")],
        "email": "Hi Jiayi, I am going to be late for our meeting tomorrow. Can we reschedule?"
    },
    config=config
)

pprint(response)
print(response["__interrupt__"])

# ======================================================= #
# ======================= Approve ======================= #
# ======================================================= #
approve_res = agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}
    ),
    config=config
)
pprint(approve_res)

# ======================================================= #
# ======================= Reject ======================== #
# ======================================================= #
reject_res = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    "message": "No please sign off."
                }
            ]
        }
    ),
    config=config
)
pprint(reject_res)

# ======================================================= #
# ========== Edit (Edit then approve immediately) ======= #
# ======================================================= #
edit_res = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    # Edited action with tool name and args
                    "edited_action": {
                        # Tool name
                        "name": "send_email",
                        # Arguments to pass to the tool
                        "args": {"body": "This is the last straw, you're fired."}
                    }
                }
            ]
        }
    ),
    config=config
)
pprint(edit_res)