from litellm import completion
import os

response = completion(
    model="groq/qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    messages=[
        {"role": "user", "content": "Hello, explain what an AI agent is."}
    ]
)

print(response.choices[0].message.content)