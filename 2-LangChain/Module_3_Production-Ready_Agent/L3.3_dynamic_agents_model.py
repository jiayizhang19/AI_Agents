from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

load_dotenv()
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
gemini_model_3 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

@wrap_model_call
def state_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on state conversation length"""
    message_count = len(request.messages)
    if message_count > 10:
        model = gemini_model_3
    else:
        model = gemini_model
    request = request.override(model=model)
    return handler(request)


agent = create_agent(
    model=gemini_model,
    middleware=[state_based_model],
    system_prompt="You are roleplaying a real life helpful office intern"
)

# ==================================================== #
# ================ Short Conversation ================ #
# ==================================================== #
response = agent.invoke(
    {"messages": [HumanMessage(content="Did you water the office plant today?")]}
)

pprint(response)
pprint(response["messages"][-1].response_metadata["model_name"])


# ==================================================== #
# ================= Long Conversation ================ #
# ==================================================== #
response_long_conv = agent.invoke(
    {"messages": [
        HumanMessage(content="Did you water the office plant today?"),
        AIMessage(content="Yes, I gave it a light watering this morning."),
        HumanMessage(content="Has it grown much this week?"),
        AIMessage(content="It's sprouted two new leaves since Monday."),
        HumanMessage(content="Are the leaves still turning yellow on the edges?"),
        AIMessage(content="A little, but it's looking healthier overall."),
        HumanMessage(content="Did you remember to rotate the pot toward the window?"),
        AIMessage(content="I rotated it a quarter turn so it gets more even light."),
        HumanMessage(content="How often should we be fertilizing this plant?"),
        AIMessage(content="About once every two weeks with a diluted liquid fertilizer."),
        HumanMessage(content="When should we expect to have to replace the pot?")
        ]}
)

pprint(response_long_conv)
print(response_long_conv["messages"][-1].response_metadata["model_name"])
