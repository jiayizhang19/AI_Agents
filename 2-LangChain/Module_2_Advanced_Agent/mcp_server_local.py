from dotenv import load_dotenv
import logging
import sys
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from typing import Any, Dict
from requests import get

load_dotenv()
tavily_client = TavilyClient()
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("./log/mcp_server")


# ========================================================= #
# =================== Create MCP Server =================== #
# ========================================================= #
# A server named "mcp_server_local" that accepts/returns any data type.
mcp = FastMCP[Any]("mcp_server_local")

# ============================================= #
# =================== Tools =================== #
# ============================================= #
@mcp.tool()
# Dict[str, Any]: A dictionary where keys are strings and values are any type.
def search_web(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    results = tavily_client.search(query=query)
    return results


# ================================================= #
# =================== Resources =================== #
# ================================================= #
@mcp.resource("github://langchain-ai/langchain-mcp-adapters/blob/main/README.md")
def github_file():
    """Resource for accessing langchain-ai/langchain-mcp-adapters/README.md file"""
    url = f"https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/blob/main/README.md"
    try:
        resp = get(url)
        return resp.text
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================== #
# =================== Prompts =================== #
# =============================================== #
@mcp.prompt()
def prompt():
    """Analyze data from a langchain-ai repo file with comprehensive insights"""
    return """
    You are a helpful assistant that answers user questions about LangChain, LangGraph and LangSmith.

    You can use the following tools/resources to answer user questions:
    - search_web: Search the web for information
    - github_file: Access the langchain-ai repo files

    If the user asks a question that is not related to LangChain, LangGraph or LangSmith, you should say "I'm sorry, I can only answer questions about LangChain, LangGraph and LangSmith."

    You may try multiple tool and resource calls to answer the user's question.

    You may also ask clarifying questions to the user to better understand their question.
    """

if __name__ == "__main__":
    mcp.run(transport="stdio")