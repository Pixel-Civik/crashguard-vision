import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="hello",
)
print(response.usage_metadata)
print(dir(response.usage_metadata))
