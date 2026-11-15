# All imports at the top
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext, Tool


# tool with dependency injection
async def fetch_data(ctx: RunContext, x: int, y: int) -> str:
    # Access dependencies via ctx.deps if needed
    return f"Fetched data for {x} and {y}"


# tool preparation for conditional availability
async def prepare_fetch_data(ctx: RunContext, tool_def: Any):
    if ctx.deps == 42:  # Example condition
        return tool_def
    return None


fetch_data_tool = Tool(fetch_data, prepare=prepare_fetch_data)


# agent context injection
@dataclass
class AgentContext:
    # Setup operations
    var1: str
    var2: int
    # Teardown operations


# example agent context injection


# Define the dependencies to be injected
@dataclass
class MyDeps:
    db_connection: str
    logger: Any


# Initialize the agent with dependency injection
my_agent = Agent(
    model="deepseek:deepseek-chat",
    deps_type=MyDeps,
)


# Define a tool that requires dependency injection
@my_agent.tool
def fetch_user_data(ctx: RunContext[MyDeps], user_id: str) -> str:
    """
    Simulates fetching user data from a database.

    Args:
        ctx (RunContext): The context that includes the dependencies.
        user_id (str): The ID of the user to fetch data for.

    Returns:
        str: Simulated user data.
    """
    # Use the database connection
    db = ctx.deps.db_connection
    # Log the operation
    ctx.deps.logger.info(f"Fetching data for user {user_id} using DB: {db}")
    # Return mock user data
    return f"User data for {user_id} from {db}"


# Instantiate the agent with dependencies
def main():
    # Mock logger and database connection
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create a context with dependencies
    context = MyDeps(
        db_connection="mock_db_connection_string",
        logger=logger,
    )

    # Run the agent with a user prompt
    user_prompt = "Fetch data for user 12345"
    result = my_agent.run_sync(
        user_prompt=user_prompt,
        deps=context,  # Pass the dependency context here
    )

    # Print the result
    print(result.output)


# Run the main function
if __name__ == "__main__":
    main()
