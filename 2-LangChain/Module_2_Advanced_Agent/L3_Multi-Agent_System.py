from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2

@tool
def fav_color() -> str:
    """Return my favorite color is green"""
    return "green"

@tool
def least_fav_color() -> str:
    """Return my least favorite color is black"""
    return "black"

math_agent = create_agent(
    model=model,
    tools=[square, square_root]
)
"""
Adding detailed system prompts because sometimes, gemini model does not wrap its final answer into
AIMessage but the ToolMessage, which is the problem in multi-agent system as the parent agent only 
reads the final AIMessage content to get the subagent's answer.
Example response:
{'messages': [HumanMessage(content='Tell me my favoriate color.', additional_kwargs={}, response_metadata={}, id='24248f95-5f1c-424d-b363-9dc1e8dd2531'),
              AIMessage(content='', additional_kwargs={'function_call': {'name': 'fav_color', 'arguments': '{}'}}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--019cdc56-f7fb-7461-ab8f-e075ffe75742-0', tool_calls=[{'name': 'fav_color', 'args': {}, 'id': '1f04fbdc-96f3-48ef-b9d1-3507a0f2313e', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 85, 'output_tokens': 10, 'total_tokens': 95, 'input_token_details': {'cache_read': 0}}),
              ToolMessage(content='green', name='fav_color', id='14cc2580-4be6-4f65-92f9-e17d41fc4555', tool_call_id='1f04fbdc-96f3-48ef-b9d1-3507a0f2313e'),
              AIMessage(content='', additional_kwargs={}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--019cdc56-fa41-7560-993a-8e07ec27773c-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 110, 'output_tokens': 0, 'total_tokens': 110, 'input_token_details': {'cache_read': 0}})]}
With the implicit instruction in the system prompt:
system_prompt="You answer questions about my favorite and least favorite colors. Use the provided tools to retrieve the answer."
"""

color_agent = create_agent(
    model=model,
    tools=[fav_color, least_fav_color],
    system_prompt="""You answer questions about my favorite and least favorite colors. 
    After calling a tool, you MUST always respond with a complete sentence.
    Summarize the tool result, never return an empty response.
    """
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


"""
Clarify the system prompt, telling the main agent it should delegate to sub-agents, not answer directly.
Example of conflicting response:
{'messages': [HumanMessage(content="What's my favorite color?", additional_kwargs={}, response_metadata={}, id='2c6a0933-0283-4b69-a393-489c67cdcd5d'),
              AIMessage(content='', additional_kwargs={'function_call': {'name': 'call_color_agent', 'arguments': '{"query": "What\'s my favorite color?"}'}}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--019cdc63-ca59-7b71-b5d2-6db914dc30fa-0', tool_calls=[{'name': 'call_color_agent', 'args': {'query': "What's my favorite color?"}, 'id': '51f4f68a-2c45-45f3-880d-10168437395a', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 122, 'output_tokens': 23, 'total_tokens': 145, 'input_token_details': {'cache_read': 0}}),
              ToolMessage(content='Your favorite color is green. ', name='call_color_agent', id='72abf427-19a0-42c0-aeb2-016748ee82c3', tool_call_id='51f4f68a-2c45-45f3-880d-10168437395a'),
              AIMessage(content="I'm sorry, I can't help you with that. My favorite color is green.", additional_kwargs={}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--019cdc63-d0d4-71e1-8bc4-c13cb7683e98-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 168, 'output_tokens': 20, 'total_tokens': 188, 'input_token_details': {'cache_read': 0}})]}
Example  of inexplicit system prompt that causes the above response:
system_prompt="You are a helpful assistant who can call subagents to do math and respond to my favorite and least favorite color" 

"""


main_agent = create_agent(
    model=model,
    tools=[call_math_agent, call_color_agent],
    system_prompt=(
        "You are a helpful assistant. You do not answer questions directly" # Stops the model from hallucinating an apology
        "For math question, delegate to call_math_agent."
        "For color preferences questions, delegate to call_color_agent."
        "Always relay the sub-agent's response back to the user exactly as received." # Stops it from rephrasing in a weird way
    )
)

cal_question = HumanMessage(content="What is the square of 4.0")
info_question = HumanMessage(content="What's my least favorite color?")
other_question = HumanMessage(content="What's my favoriate food?")

response = main_agent.invoke(
    {
        "messages": [cal_question, info_question, other_question]
    }
)

pprint(response)
