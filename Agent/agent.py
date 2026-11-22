# Create the AI agent

# This is where the agentic behaviour happens.

# The agent will:
# 1. Receive a user question.
# 2. Send the question and available tool definitions to the LLM.
# 3. Let the LLM decide which tool to invoke.
# 4. Execute the requested MCP tool.
# 5. Send tool results back to the LLM.
# 6. Repeat until the LLM produces a final answer.

from multiprocessing.dummy.connection import Client
import os
import json
import asyncio

# from asyncio import tools
from litellm import completion
from fastmcp import client          

from dotenv import load_dotenv
from pydantic import ValidationError, Field

from schemas import (
    GetWeatherInput, 
    GetExchangeRateInput,
    WeatherData,
    CurrencyExchangeData,
    ToolObservation,
    LLMIterationInput,
    AgentStep, 
    AgentResponse
)

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

def convert_mcp_tools():
    """Convert MCP tool definitions into a format suitable for the LLM."""
    mcp_tools = MCP_TOOLS
    converted_tools = []
    for tool in mcp_tools:
        converted_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        }
        converted_tools.append(converted_tool)
    return converted_tools

# Create reusable tool input and output validation
def validate_tool_input(tool_name: str, parameters: dict) -> dict:
    """Validate the input parameters for a given tool."""
    if tool_name == "get_weather":
        try:
            validated_input = GetWeatherInput.model_validate(**parameters)
        except ValidationError as e:
            raise ValueError(f"Invalid input for get_weather: {e}")

    elif tool_name == "get_exchange_rate":
        try:
            validated_input = GetExchangeRateInput.model_validate(**parameters)
        except ValidationError as e:
            raise ValueError(f"Invalid input for get_exchange_rate: {e}")

    else:
        raise ValueError(f"Unknown tool: {tool_name}")
    return validated_input.model_dump()  # Return the validated input as a dictionary

def validate_tool_output(tool_name: str, output: dict) -> dict:
    """Validate the output of a given tool."""
    if tool_name == "get_weather":
        try:
            validated_output = WeatherData.model_validate(**output)
        except ValidationError as e:
            raise ValueError(f"Invalid output for get_weather: {e}")

    elif tool_name == "get_exchange_rate":
        try:
            validated_output = CurrencyExchangeData.model_validate(**output)
        except ValidationError as e:
            raise ValueError(f"Invalid output for get_exchange_rate: {e}")

    else:
        raise ValueError(f"Unknown tool: {tool_name}")
    return validated_output.model_dump()  # Return the validated output as a dictionary

# Execute an MCP tool by sending a request to the MCP server and returning the result.
async def execute_mcp_tool(tool_name: str, parameters: dict) -> dict:
    """Execute an MCP tool with input and output validation."""
    # Validate the input parameters
    validated_parameters = validate_tool_input(tool_name, parameters)

    async with client.Client(MCP_SERVER) as mcp_client:
        result = await mcp_client.invoke(tool_name, validated_parameters)

    # Validate the output result
    validated_result = validate_tool_output(tool_name, result)
    return validated_result

# Tool calling Loop using LLM to decide which tool to invoke based on user input and tool definitions.
async def tool_calling_loop(user_question: str, MCP_TOOLS):
    message_history = [
        {"role": "system", "content": "You are an AI agent that can call tools to answer user questions."},
        {"role": "user", "content": user_question}
    ]

    # Store validated tool observations.
    observations = []

    # Store tools successfully executed.
    tools_used = []

    # Set maximum number of LLM iterations.
    max_iterations = 3

    # Initialize the LLM with the Gemini API key and model
    for iteration in range(1, max_iterations + 1):  # Limit the number of iterations to avoid infinite loops
        print(f"\n[INFO] Iteration {iteration} of {max_iterations}")

        # Create Pydantic input for this iteration.
        iteration_input = LLMIterationInput(
            user_input=user_question,
            iteration_number=iteration,
            max_iterations=max_iterations,
            tool_observations=observations
        )

         # Serialize validated Pydantic input to JSON.
        iteration_payload = iteration_input.model_dump_json(
            indent=2
        )
        print(f"[DEBUG] Iteration Input:\n{iteration_payload}")

        #Pass iteration context to the LLM as a user message.
        message_history.append({"role": "user", "content": ("Process this validated iteration context" + iteration_payload)})
        # Send the user question and tool definitions to the LLM
        try:
            print("\n[INFO] Sending request to Groq...")

            print("[DEBUG] Tools being sent:")
            print(json.dumps(MCP_TOOLS, indent=2, default=str))
             # Iterations 1 and 2 may call MCP tools.
            if iteration < max_iterations:
                response = completion(
                    model=MODEL,
                    api_key=GROQ_API_KEY,
                    messages=message_history,
                    tools=MCP_TOOLS,
                    tool_choice="auto",
                    max_tokens=100,
                    temperature=0,
                )
            else:
                # The final iteration should produce a final answer without calling any tools.
                message_history.append({"role": "system", 
                                        "content": """
                                        This is the final iteration.
                                        Do not call any tools.
                                        Use the validated observations already available.
                                        Return only a JSON object matching this structure:

                                        {
                                        "status": "success",
                                        "answer": "Your final answer",
                                        "tools_used": ["weather_tool"],
                                        "iterations_used": 3
                                        }

                                        Status must be success, partial or failed.
                                        Do not invent missing information.
                                        """
                                    })
                response = completion(
                    model=MODEL,
                    api_key=GROQ_API_KEY,
                    messages=message_history,
                    max_tokens=500,
                    temperature=0,
                    response_format={
                        "type": "json",
                        "schema": AgentResponse.model_json_schema()
                    }
                )

            print("[SUCCESS] Groq response received!")
            print(response)
        except Exception as e:
            print(f"Error occurred while calling LLM: {str(e)}")
            break

        # Get the LLM's response
        llm_message = response["choices"][0]["message"]
        # Convert the assistant message into a dictionary.
        assistant_message = message_to_dict(llm_message)
        message_history.append(assistant_message)

        # Extract tool calls using Groq's standard format.
        tool_calls = getattr(
            llm_message,
            "tool_calls",
            None,
        )

      
        # STEP 6: Handle final response.
        # --------------------------------------------------

        if not tool_calls:

            # Extract LLM response content.
            content = llm_message.content
            try:
                final_json = json.loads(content)

                # Validate final response using Pydantic.
                final_response = AgentResponse.model_validate(final_json)
                print("\n[INFO] Final response from LLM:")
                print(final_response.model_dump_json(indent=2))
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse final response as JSON: {str(e)}")
                break

        # STEP 7: Execute MCP tool calls.
        for tool_call in tool_calls:
            tool_name = tool_call.functionname
            parameters = tool_call.function.parameters

            try:
                if isinstance(parameters, dict):
                    parameters = json.loads(parameters)
                    # Validate tool input parameters using Pydantic.
                    validated_parameters = validate_tool_input(tool_name, parameters)
                else:
                    parameters = parameters

                print(
                    f"\n[INFO] Executing MCP tool: {tool_name}"
                )

                print("[DEBUG] Raw tool arguments:")
                print(parameters)

                # Execute MCP tool.
                # This validates arguments and result internally.

                print(f"\n[INFO] LLM requested tool: {tool_name} with parameters: {parameters}")

                # Execute the requested MCP tool
                
                tool_result = await execute_mcp_tool(tool_name, parameters)
                print(f"[SUCCESS] Tool {tool_name} executed successfully. Result: {tool_result}")
                observation = ToolObservation(
                        tool_name=tool_name,
                        success=True,
                        data=tool_result
                    )
                # Store the validated observation.
                observations.append(observation)
                
                # Track successful tool usage.
                if tool_name not in tools_used:
                    tools_used.append(tool_name)

                # Send successful result back to Groq.
                tool_content = json.dumps(
                    observation.model_dump(mode="json")
                )

                print("[SUCCESS] MCP tool executed")
   
             
            except Exception as e:
                print(f"[ERROR] Error executing tool {tool_name}: {str(e)}")
                message_history.append({"role": "tool", "content": f"Error executing tool {tool_name}: {str(e)}"})

    # STEP 8: Handle exhausted iteration budget.
    # --------------------------------------------------

    raise RuntimeError(
        "Agent exhausted all three iterations "
        "without producing a valid final response"
    )


        # # Check if the LLM has produced a final answer
        # if llm_message["role"] == "assistant" and "final_answer" in llm_message:
        #     print("Final Answer:", llm_message["final_answer"])
        #     break
            
        # # Get the LLM's response
        # llm_message = response["choices"][0]["message"]
        # message_history.append(llm_message)

        # # Check if the LLM has produced a final answer
        # if llm_message["role"] == "assistant" and "final_answer" in llm_message:
        #     print("Final Answer:", llm_message["final_answer"])
        #     break

        # # If the LLM wants to call a tool, extract the tool name and parameters
        # if llm_message["role"] == "assistant" and "tool_call" in llm_message:
        #     tool_call = llm_message["tool_call"]
        #     tool_name = tool_call["name"]
        #     parameters = tool_call.get("parameters", {})

        #     # Execute the requested MCP tool
        #     try:
        #         tool_result = await execute_mcp_tool(tool_name, parameters)
        #         message_history.append({"role": "tool", "content": json.dumps(tool_result)})
        #     except Exception as e:
        #         message_history.append({"role": "tool", "content": f"Error executing tool {tool_name}: {str(e)}"})

if __name__ == "__main__":
    user_question = input("Enter your question: ")
    if MODEL.startswith("groq"):
        MCP_TOOLS = convert_mcp_tools()  # Convert MCP tools for Groq model
    result = asyncio.run(tool_calling_loop(user_question, MCP_TOOLS))

    # Display validated final JSON.
    print("\n========== FINAL AGENT RESPONSE ==========")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )