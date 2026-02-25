from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient()

gemini_model = "gemini-3-flash-preview"
question = "Who is the current mayor of San Francisco?"

model = ChatGoogleGenerativeAI(model=gemini_model)
question_to_agent = HumanMessage(content=question)

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)
# web_search.invoke("Who is the current mayor of San Francisco?")

agent = create_agent(
    model=model,
    tools=[web_search]
)

response = agent.invoke(
    {"messages": [question_to_agent]}
)

pprint(response["messages"][-1].content[0]["text"])
