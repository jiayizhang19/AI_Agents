from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.messages import AIMessage

load_dotenv()

# ================================================================ #
# =================== Initialize a Chatbot ======================= #
# ================================================================ #
gemini_model = "gemini-3-flash-preview"
model = ChatGoogleGenerativeAI(model=gemini_model)
response = model.invoke("What's the captical of the Moon?")
print(response.text)

# ================================================================ #
# =================== Initialize an Agent ======================== #
# ================================================================ #
# agent_model = "gemini-3.1-pro-preview" # This will direct to vertexai instead of gemini model
agent = create_agent(model=model)
# Interact with the agent just using the "invoke"
response = agent.invoke(
    {"messages":[HumanMessage(content="What's the captical of the Moon?")]}
)
# pprint(response)
pprint(response["messages"][-1].content[0]["text"])

response_to_ai_message = agent.invoke(
    {"messages": [
        HumanMessage(content="What's the captical of the Moon?"),
        AIMessage(content="The captical of the Moon is Luna City"),
        HumanMessage(content="Interesting, tell me more about Luna City")
    ]}
)
# pprint(response_to_ai_message)
pprint(response_to_ai_message["messages"][-1].content[0]["text"])