from litellm import completion
import os

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string"
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            }
        }
    }
]

response = completion(
    model="groq/qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    messages=[
        {"role": "user", "content": "What is the weather in Hyderabad?"}
    ],
    tools=tools,
    tool_choice="auto"
)

print(response.choices[0].message.content)