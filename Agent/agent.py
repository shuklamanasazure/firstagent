# Create the AI agent

# This is where the agentic behaviour happens.

# The agent will:
# 1. Receive a user question.
# 2. Send the question and available tool definitions to the LLM.
# 3. Let the LLM decide which tool to invoke.
# 4. Execute the requested MCP tool.
# 5. Send tool results back to the LLM.
# 6. Repeat until the LLM produces a final answer.

import os
import json
import asyncio

from litellm import completion
from fastmcp import client          

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Gemini API key from environment variable
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# MODEL = "gemini/gemini-3.8-flash"    # Model to use for the LLM   
MODEL = "groq/qwen/qwen3.8-27b"    # Model to use for the LLM

MCP_SERVER = "http://127.0.0.1:8000/mcp"  # MCP server URL

# MCP Tools definition exposed to the LLM. The LLM will use these definitions to decide which tool to invoke.
MCP_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location for which to get weather"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_exchange_rate",
        "description": "Get the exchange rate for a currency pair",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "The amount to convert"
                },
                "base_currency": {
                    "type": "string",
                    "description": "The base currency code (e.g., USD)"
                },
                "target_currency": {
                    "type": "string",
                    "description": "The target currency code (e.g., EUR)"
                }
            },
            "required": ["amount", "base_currency", "target_currency"]
        }
    }   
]

def convert_mcp_tools(mcp_tools):
    """Convert MCP tool definitions into a format suitable for the LLM."""
    converted_tools = []
    for tool in mcp_tools:
        converted_tool = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"]
        }
        converted_tools.append(converted_tool)
    return converted_tools



# Execute an MCP tool by sending a request to the MCP server and returning the result.
async def execute_mcp_tool(tool_name: str, parameters: dict) -> dict:
    async with client.Client(MCP_SERVER) as mcp_client:
        result = await mcp_client.invoke(tool_name, parameters)
        # convert result into JSON text for better readability
        return result

# Tool calling Loop using LLM to decide which tool to invoke based on user input and tool definitions.
async def tool_calling_loop(user_question: str):
    message_history = [
        {"role": "system", "content": "You are an AI agent that can call tools to answer user questions."},
        {"role": "user", "content": user_question}
    ]
    # Initialize the LLM with the Gemini API key and model
    for iteration in range(3):  # Limit the number of iterations to avoid infinite loops
        # Send the user question and tool definitions to the LLM
        response = completion(
            model=MODEL,
            messages=message_history,
            tools=MCP_TOOLS,
            tool_choice="auto",
            max_tokens=100
        )
        # Get the LLM's response
        llm_message = response["choices"][0]["message"]
        message_history.append(llm_message)

        # Check if the LLM has produced a final answer
        if llm_message["role"] == "assistant" and "final_answer" in llm_message:
            print("Final Answer:", llm_message["final_answer"])
            break

        # If the LLM wants to call a tool, extract the tool name and parameters
        if llm_message["role"] == "assistant" and "tool_call" in llm_message:
            tool_call = llm_message["tool_call"]
            tool_name = tool_call["name"]
            parameters = tool_call.get("parameters", {})

            # Execute the requested MCP tool
            try:
                tool_result = await execute_mcp_tool(tool_name, parameters)
                message_history.append({"role": "tool", "content": json.dumps(tool_result)})
            except Exception as e:
                message_history.append({"role": "tool", "content": f"Error executing tool {tool_name}: {str(e)}"})

if __name__ == "__main__":
    user_question = input("Enter your question: ")
    if MODEL.startswith("groq"):
        MCP_TOOLS = convert_mcp_tools(MCP_TOOLS)  # Convert MCP tools for Groq model
    asyncio.run(tool_calling_loop(user_question))  