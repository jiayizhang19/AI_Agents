from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from tavily import TavilyClient
from langchain_mcp_adapters.client import MultiServerMCPClient

# ============================================ #
# One coordinator agent with three sub-agents:
# 1. Flight agent --> MCP Server
# 2. Venue agent --> Web Search
# 3. Music playlist agent --> SQL Query
# ============================================ #


load_dotenv()
model = ChatGoogleGenerativeAI("gemini-2.5-flash")
tav_client = TavilyClient()
MCP_client = MultiServerMCPClient(
    {
        "travel_server":{
            "url": "https://mcp.kiwi.com",
            "transport": "streamable_http"
        }
    }
)

@tool 
def search_for_filghts(query: str) -> str:
    """
    Search available flight tickets on the web, and return flight, date, price info to users.
    """
    return tav_client(query)

@tool
def venue():
    """
    Search for 
    """