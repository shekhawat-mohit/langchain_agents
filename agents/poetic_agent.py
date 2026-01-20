from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()   

messages = [{"role": "user", "content": "Write a poem about the sea."},
            {"role": "system", "content": "You are a nature poet who writes poems in 4 lines."}]

agent = create_agent(
    model="openai:gpt-5-mini",
)
# result = agent.invoke(
#     {"messages": messages})

# for message in result["messages"]:
#     message.pretty_print()

for token, metadata in agent.stream(
    {"messages": messages},
    stream_mode="messages",
):
    print(token, metadata)