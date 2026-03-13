from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from typing import Dict, Any
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentState
from langgraph.types import Command
from datetime import date

# ============================================ #
# One coordinator agent with three sub-agents:
# 1. Flight agent --> MCP Server
# 2. Venue agent --> Web Search
# 3. Music playlist agent --> SQL Query
# ============================================ #


load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
tav_client = TavilyClient()
db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")
today = date.today().strftime("%Y-%m-%d")

# ================================================== #
# ================== Create Tools ================== #
# ================================================== #
@tool 
def web_search(query: str) -> Dict[str, Any]:
    """
    Search the web for information
    """
    return tav_client.search(query)

@tool
def query_playlist_db(query: str) -> str:
    """
    Query the database for playlist information
    """
    try:
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"
    
# ================================================== #
# ================== Create State ================== #
# ================================================== #
class WeddingState(AgentState):
    origin: str
    destination: str
    Date: str
    guest_count: str
    genre: str


# ================================================== #
# ================ Create Subagents ================ #
# ================================================== #
flight_client = MultiServerMCPClient(
    {
        "travel_server":{
            "url": "https://mcp.kiwi.com",
            "transport": "streamable_http"
        }
    }
)
async def travel_agent():
    tools = await flight_client.get_tools()
    travel_agent = create_agent(
        model=model,
        tools = tools,
        system_prompt=f"""
            You are a travel agent. Search for flights to the desired destination wedding location.
            You are not allowed to ask any more follow up questions. 
            You must find the best flight options based on the following:
            - Price (lowest, economic class)
            - Duration (shortest)
            - Date (only search for flights after {today})
            To make things easy, only look for one way direct ticket.
            You may need to do multiple searches to iteratively find the best options.
            You will be given no extra information, only the origin and the destination. 
            It is your job to think critically about it. Once you have found the best options, let the user know your shortlist of options. 
        """
    ) 
    return travel_agent

venue_agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt="""
        You are a venu agent. Search for venues in the desired location, and with the desired capacity.
        You must find the best venue options based on the following:
        - Price (lowest)
        - Capacity (exact match)
        - Reviews (highest)
        You may need to make multiple searches to iteratively find the best options.
    """
)

playlist_agent = create_agent(
    model=model,
    tools=[query_playlist_db],
    system_prompt="""
        You are a playlist agent. Query the sql database and curate the perfect playlist for a wedding given a genre.
        Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
        If you run into errors when querying the database, try to fix them by making changes to the query.
        If a query fails more than 5 times, return the best results you have gotten so far.
    """
)

# ================================================== #
# ====== Create Agent Call Tools and Coordinator === #
# ================================================== #
@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """
    Travel agent searches for flights to the desired destination wedding location.
    """
    agent = await travel_agent()
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    date = runtime.state["Date"]
    response = await agent.ainvoke(
        {
            "messages": HumanMessage(content=f"Find flights from {origin} to {destination} on or before {date}")
        }
    )
    return response["messages"][-1].content

@tool
def search_venues(runtime: ToolRuntime) -> str:
    """
    Venue agent chooses best venue for the given location and capacity. 
    """
    destination = runtime.state["destination"]
    capacity = runtime.state["guest_count"]
    date = runtime.state["Date"]
    query = f"Find wedding venues in {destination} for {capacity} guests on {date}"
    response = venue_agent.invoke(
        {
            "messages": HumanMessage(content=query)
        }
    )
    return response["messages"][-1].content

@tool
def suggest_playlist(runtime: ToolRuntime) -> str:
    """
    Playlist agent curates the perfect playlist for the given genre
    """
    genre = runtime.state["genre"]
    query = f"Find {genre} tracks for wedding playlist"
    response = playlist_agent.invoke(
        {
            "messages": HumanMessage(content=query)
        }
    )
    return response["messages"][-1].content

@tool
def update_state(origin: str, destination: str, date: str, guest_count: str, genre: str, runtime: ToolRuntime) -> str:
    """
    Update the state when you know all the values: origin, destination, date, guest_count, genre
    """
    return Command[tuple[()]](
        update={
            "origin": origin,
            "destination": destination,
            "Date": date,
            "guest_count": guest_count,
            "genre": genre,
            "messages": [ToolMessage(
                content="Successfully updated state",
                tool_call_id=runtime.tool_call_id
            )]
        }
    )


coordinator = create_agent(
    model=model,
    tools=[search_flights, search_venues, suggest_playlist, update_state],
    state_schema=WeddingState,
    system_prompt="""
        You are a wedding coordinator. Delegate tasks to your specialists for flights, venues and playlists.
        You must follow these steps in strict order:
        1. First, find out the wedding details: origin, destination, guest count and
        music genre from the user message. 
        2. Call update_state tool to update the state with all the details you have found. 
        Do not skip this step.
        3. Only after call update_state, you can call the other three tools. 
        Once you have received their answers, coordinate the perfect wedding for me.
    """
)

# ================================================== #
# =============== Multi-agent system =============== #
# ================================================== #
query = "I'm from London and I'd like a wedding in Paris for 100 guests on April 15th 2029, jazz-genre"
async def call_coordinator():
    response = await coordinator.ainvoke(
        {
            "messages": HumanMessage(content=query)
        }
    )
    pprint(response["messages"][-1].content)

asyncio.run(call_coordinator())