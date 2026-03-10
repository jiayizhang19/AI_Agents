from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from pprint import pprint

load_dotenv()

# ========================================================== #
# ==================== Tool Definition ===================== #
# ========================================================== #
# By default, the tool's name comes from the function name.
# Ensure the name and description(docstring) are detailed enough for an agent to understand.
@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

# ========================================================== #
# ====================== Custom Tool ======================= #
# ========================================================== #
@tool("square_root")
def customed_square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5
# Write description besides tool decrator rathar than in the docstring 
@tool("square_root", description="Calculate the square root of a number")
def customed_square_root(x: float) -> float:
    return x ** 0.5

# ========================================================== #
# ============== Invoke Tool without an Agent ============== #
# ========================================================== #
square_root.invoke({"x": 121})

# ========================================================== #
# ============== Invoke Tool within an Agent =============== #
# ========================================================== #
gemini_model = "gemini-3-flash-preview"
model = ChatGoogleGenerativeAI(model=gemini_model)
cal_agent = create_agent(
    model=model,
    system_prompt="You are an arithmetic wizard. Use your tools to calculate the square root and square of any number",
    tools=[square_root]
)
response = cal_agent.invoke(
    {"messages": HumanMessage(content="What's the square root of 467?")}
)
pprint(response["messages"][-1].content[0]["text"])