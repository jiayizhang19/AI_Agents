from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of x"""
    return x ** 2

# @tool
# def fav_color() -> str:
#     """Tell others my favoriate color"""
#     return "green"

# @tool
# def least_fav_color() -> str:
#     """Tell other my least favorite color"""
#     return "black"

square_agent = create_agent(
    model=model,
    tools=[square]
)

square_root_agent = create_agent(
    model=model,
    tools=[square_root]
)

@tool
def call_square_agent(x: float) -> float:
    """Call sub-agent square agent to calculate square on the given input."""
    response = square.invoke(
        {"messages": [HumanMessage(content="Calculate the square of {x}")]}
    )
    return response["messages"][-1].content


@tool
def call_square_root_agent(x: float) -> float:
    """Call sub-agent square_root_agent to calculate square root on the given input."""
    response = square.invoke(
        {"messages": [HumanMessage(content="Calculate the square root of {x}")]}
    )
    return response["messages"][-1].content

# @tool
# def call_info_agent(question) -> str:
#     """Call sub-agent info_agent to answer my favoriate and least favoriate color."""
#     response = info_agent.invoke(
#         {"messages": [HumanMessage(content=question)]}
#     )
#     return response["messages"][-1].content


main_agent = create_agent(
    model=model,
    tools=[call_square_agent, call_square_root_agent],
    system_prompt="You are a helpful assistant who can call subagents to calculate square and square root of given input" 
)

question = HumanMessage(content="What is square root of 4.0")

response = main_agent.invoke(
    {
        "messages": [question]
    }
)

pprint(response)
