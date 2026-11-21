from litellm import completion
import os

response = completion(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    messages=[
        {"role": "user", "content": "Hello, explain what an AI agent is."}
    ]
)

print(response.choices[0].message.content)