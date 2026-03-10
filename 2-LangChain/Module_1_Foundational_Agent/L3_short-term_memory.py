from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
# Short-Term Memory
from langgraph.checkpoint.memory import InMemorySaver 


load_dotenv()
gemini_model = "gemini-3-flash-preview"
question = HumanMessage(content="Hello my name is Jiayi and my favourite color is green")
followup_question = HumanMessage(content="What's my favourite color?")

# =================================================================================== #
# ============= Define Thread ID to Group Checkpointers of State Together =========== #
# =================================================================================== #
config = {"configurable": {"thread_id": "1"}}
agent = create_agent(
    model=ChatGoogleGenerativeAI(model=gemini_model),
    checkpointer=InMemorySaver()
)

# =================================================================================== #
# ======================= Invoke Agents with Common Thread ID ======================== #
# =================================================================================== #
response = agent.invoke(
    {"messages": [question]},
    config=config
)
pprint(response)
# Literally retains memory of the previous conversation and appended it to its list of messages.
response = agent.invoke(
    {"messages": [followup_question]},
    config=config
)
pprint(response)