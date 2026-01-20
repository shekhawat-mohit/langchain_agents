from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import nest_asyncio, asyncio

#nest_asyncio.apply()

load_dotenv()

async def async_mcp_call() -> list:
    mcp_client = MultiServerMCPClient(
        {
            "time":{
                "transport":"stdio",
                "command":"npx",
                "args":["-y","@theo.foobar/mcp-time"]
            }
        },
    )

    #Load tools from the MCP server
    mcp_tools = await mcp_client.get_tools()
    print(f"Loaded {len(mcp_tools)} tools from MCP server.")
    for tool in mcp_tools:
        print(f"- {tool.name}")
    return mcp_tools

async def create_agent() -> object:    
    mcp_tools = await async_mcp_call()
    print("Creating agent with MCP tools...")
    agent = create_agent(
    model="openai:gpt-5-mini",
    tools=mcp_tools,
    system_prompt="You are an expert time-telling agent. Use the available tools to provide accurate time information.",
    )
    return agent

async def main():
    agent = await create_agent()
    result = await agent.async_invoke(
        {"messages": [{"role": "user", "content": "What time is it in Tokyo right now?"}]}
    )

    for message in result["messages"]:
        message.pretty_print()

asyncio.run(main())

# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())




