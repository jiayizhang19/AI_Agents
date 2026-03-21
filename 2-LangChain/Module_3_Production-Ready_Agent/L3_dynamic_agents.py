from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest


load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@dataclass
class LanguageContext:
    user_language: str = "English"

@dynamic_prompt
def user_language_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_language = request.runtime.context.user_language
    # First Prompt: telling the agent to respond in English
    base_prompt = "You are a helpful assistant." 
    # Second Prompt: telling it to respond in whatever language is in the context
    if user_language != "English":
        return f"{base_prompt} only respond in {user_language}."
    elif user_language == "English":
        return base_prompt
    

agent = create_agent(
    model=model,
    context_schema=LanguageContext,
    middleware=[user_language_prompt]
)


response = agent.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="Chinese")
)

pprint(response)
