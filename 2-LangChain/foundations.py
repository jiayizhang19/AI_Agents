from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
response = model.invoke("What's the captical of the Moon?")
print(response.content)