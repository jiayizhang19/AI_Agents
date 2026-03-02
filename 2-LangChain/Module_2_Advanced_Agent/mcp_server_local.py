from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from typing import Any, Dict
from requests import get

load_dotenv()
tavily_client = TavilyClient()

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

# =============================================== #
# =================== Prompts =================== #
# =============================================== #



# ================================================= #
# =================== Resources =================== #
# ================================================= #
