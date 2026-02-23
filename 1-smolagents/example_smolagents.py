# CodeAgent() provides access to Hugging Face's severless inference API. 
# Use DuckDuckGoSearchTool to search the web.
# Use @tool to create custom tools with specific business logic
# Use additional_authorized_import to allow code generation using other libraries.



from huggingface_hub import login
from dotenv import load_dotenv
import os
from smolagents import tool, Tool, CodeAgent,ToolCallingAgent, DuckDuckGoSearchTool, InferenceClientModel, LiteLLMModel
import litellm
import time
import datetime


# ======================================================================================== #
# =========================== 1.1 Custom New Tools: Decorator ============================ #
# ======================================================================================== #
@tool
def suggest_menu(occasion: str) -> str:
    # Keep an eye on this docstring written, it needs to have description for all arguments (with the same upper or lower case).
    # Otherwise, the tool is not going to work.
    """
    Suggests a menu based on the occasion.
    Args:
        occasion(str): The type of occasion for the party. Allowed values are:
            - "causal": Menu for causal party.
            - "formal": Menu for formal party.
            - "superhero": Menu for superhero party.
            - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks"
    if occasion == "formal":
        return "3-course dinner with wine and dessert"
    if occasion == "superhero":
        return "Buffet with high-energy and healthy food"
    else:
        return "Custome menu for the butler"

# ======================================================================================== #
# =========================== 1.2 Custom New Tools: Subclass ============================ #
# ======================================================================================== #
class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""
    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }
    output_type = "string"

    def forward(self, category:str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }
        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")

# ======================================================================================== #
# ================================= 2. LLM Integration =================================== #
# ======================================================================================== #
load_dotenv()
# Switching from HF inference model to free tier gemini model
# HF_TOKEN = os.getenv("HF_TOKEN")
# login(token=HF_TOKEN)

# litellm._turn_on_debug()
gemini_model = LiteLLMModel(
    # model_id="gemini/gemini-2.5-flash-lite",
    model_id="gemini/gemini-3-flash-preview",
    api_key=os.getenv("GEMINI_API_KEY")
)

# ======================================================================================== #
# ================ 3.1.1 Code Agent: Using web search and custom tools =================== #
# ======================================================================================== #
web_and_tools_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(),suggest_menu],
    model=gemini_model,
    # model=InferenceClientModel(), # It needs credits
)

web_and_tools_agent.run(
    "Search for the best music recommendations for a party."
    "Prepare me a menu for a party that is informal."
)


# ======================================================================================== #
# === 3.1.2 Code Agent: Using Python imports inside the agent (agent generation code) ==== #
# ======================================================================================== #
python_imports_agent = CodeAgent(
    tools=[],
    # model=InferenceClientModel(),
    model=gemini_model,
    additional_authorized_imports=["datetime"]
)

python_imports_agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """
)

# ======================================================================================== #
# =============== 3.2 Tool Calling Agent: Using web search and custom tools ============== #
# ======================================================================================== #
search_tool = DuckDuckGoSearchTool()
party_theme_tool = SuperheroPartyThemeTool()
tool_calling_agent = ToolCallingAgent(
    model=gemini_model,
    tools=[
        search_tool,
        suggest_menu, 
        party_theme_tool
    ],
)

tool_calling_agent.run(
    "Search for the best music recommendations for a party."
    "Prepare me a menu for a party that is informal."
    "What would be a good superhero party idea for a 'villain masquerade' theme."
)
