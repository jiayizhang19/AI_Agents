from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

# MCP available https://mcp.so/server/kiwi-travel-mcp/Vytautas%20Dargis

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

client = MultiServerMCPClient(
    {
        "travel_server":{
            "transport": "streamable_http",
            "url": "https://mcp.kiwi.com"
        }
    }
)

async def build_agent():
    tools = await client.get_tools()
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt="You are a travel agent."
    )
    return agent

    
async def execute_agent(agent):
    question = HumanMessage(content="Get me flights from San Francisco to New York on 31st March, 2026.")
    config = {"configurable": {"thread_id": "1"}}
    response = await agent.ainvoke(
        {"messages": [question]},
        config
    )
    return response["messages"][-1].content[0]["text"]

async def main():
    agent = await build_agent()
    response = await execute_agent(agent)
    pprint(response)

if __name__ == "__main__":
    asyncio.run(main())