import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


model = init_chat_model(
    "gpt-4.1",
    api_key=os.getenv(('OPENAI_API_KEY'))
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
result = agent.invoke({"messages": [
    {
        "role": "user",
        "content": "what is the weather in sf"
    }
]}
)

# 2. Extract the content from the last message
# The messages key contains the full history of the turn
final_message = result["messages"][-1]

# 3. Print the text content
print(final_message.content)
