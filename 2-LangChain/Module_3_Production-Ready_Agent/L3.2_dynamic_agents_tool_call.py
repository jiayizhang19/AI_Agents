from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from dataclasses import dataclass
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from tavily import TavilyClient
from langchain_community.utilities import SQLDatabase
from typing import Dict, Any, Callable


load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tavily_client = TavilyClient()
db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

@tool
def sql_query(query: str) -> str:
    """Obtain information from the database using SQL queries"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

@dataclass
class UserRole:
    user_role: str = "external"


@wrap_model_call
def dynamic_tool_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Dynamically call tools based on the runtime context"""
    user_role = request.runtime.context.user_role
    if user_role == "internal":
        pass # internal users get access to all tools
    else:
        tools = [web_search]
        request = request.override(tools=tools)
    return handler(request) # must call LLM here, you decide when to call it and with what request

agent = create_agent(
    model=model,
    tools=[web_search, sql_query],
    middleware=[dynamic_tool_call],
    context_schema=UserRole
)

response = agent.invoke(
    {"messages": HumanMessage(content="How many artists are in the database?")},
    context={"user_role": "internal"}
)

pprint(response)