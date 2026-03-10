from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of x"""
    return x ** 2

@tool
def fav_color() -> str:
    """Return my favorite color"""
    return "green"

@tool
def least_fav_color() -> str:
    """Return my least favorite color"""
    return "black"

math_agent = create_agent(
    model=model,
    tools=[square, square_root]
)

color_agent = create_agent(
    model=model,
    tools=[fav_color, least_fav_color],
    system_prompt="You answer questions about my favorite and least favorite colors. Use the provided tools to retrieve the answer."
)

@tool
def call_math_agent(query: str) -> str:
    """Call math agent to perform a calculation described in the query."""
    response = math_agent.invoke(
        {"messages": [HumanMessage(content=query)]}
    )
    return response["messages"][-1].content


@tool
def call_color_agent(query: str) -> str:
    """Call color agent to return my favorite and least favorite color."""
    response = color_agent.invoke(
        {"messages": [HumanMessage(content=query)]}
    )
    return response["messages"][-1].content


main_agent = create_agent(
    model=model,
    tools=[call_math_agent, call_color_agent],
    system_prompt="You are a helpful assistant who can call subagents to do math and respond to my favorite and least favorite color" 
)

cal_question = HumanMessage(content="What is the square root of 4.0")
info_question = HumanMessage(content="What's my favorite color?")

response = main_agent.invoke(
    {
        "messages": [cal_question, info_question]
    }
)

pprint(response)
