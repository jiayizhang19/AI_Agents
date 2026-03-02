from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint

client = MultiServerMCPClient(
    {
        "local_server": {
            "transport": "stdio",
            # "command": "python",
            "command": r"d:\program_envs\ai_agents\Scripts\python.exe",
            "args": ["mcp_server_local.py"],
        }
    }
)

async def build_agent():
    try:
        tools = await client.get_tools()
        resources = await client.get_resources("local_server")
        prompt = await client.get_prompt("local_server", "prompt")
        prompt = prompt[0].content

        agent = create_agent(
            model=ChatGoogleGenerativeAI("gemini-2.5-flash"),
            tools=tools,
            system_prompt=prompt,
        )

        return agent
    except Exception as e:
        print(f"Error: {e}")
        raise e

async def run_agent(agent):
    config = {"configurable": {"thread_id": "1"}}
    response = agent.invoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
        config=config
    )    
    return response

async def main():
    agent = await build_agent()
    response = await run_agent(agent)
    pprint(response)

if __name__ == "__main__":
    asyncio.run(main())