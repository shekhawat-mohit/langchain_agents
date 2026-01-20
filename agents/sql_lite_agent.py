from dataclasses import dataclass
from unittest import result
from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
import os

load_dotenv()

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///agents/config/Chinook.db")

# define context structure to support dependency injection
# this will allow us to pass in different database connections if needed
# Its not needed here since this is Chinook database
@dataclass
class AgentContext:
    db: SQLDatabase
    is_employee: bool

@tool
def run_sql_query(runtime: ToolRuntime[AgentContext],query: str) -> str:
    """Run a SQL query against the connected database and return the results."""
    db = runtime.context.db
    try:
        result = db.run(query)
        return str(result)
    except Exception as e:
        return f"Error executing query: {str(e)}"
    
SYSTEM_PROMPT = """You are an expert SQL agent. 
Rules:
- Think step by step.
- When you need data, use the tool 'run_sql_query' with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows per query.
{table_limits}
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column names over SELECT *."""

from langchain.agents.middleware import ModelRequest, dynamic_prompt

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    
    if not request.runtime.context.is_employee:
        table_limits = "- Limit access to these tables: Album, Artist, Genre, Playlist, PlaylistTrack, Track."
    else:
        table_limits = ""
    return SYSTEM_PROMPT.format(table_limits=table_limits)

from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[run_sql_query],
    middleware=[dynamic_system_prompt],
    system_prompt=SYSTEM_PROMPT,
    context_schema=AgentContext,
    checkpointer=InMemorySaver(),
)

question = "This is Frank Harris. What was the total on my last invoice?"

for step in agent.stream(
    {"messages": question},
    {"configurable":{"thread_id":"1"}},
    context=AgentContext(db=db, is_employee=True),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

question = "What were the titles?"

for step in agent.stream(
    {"messages": question},
    {"configurable":{"thread_id":"1"}},
    context=AgentContext(db=db, is_employee=True),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

    


