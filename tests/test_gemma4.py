"""Quick Gemma 4 validation test."""
from dotenv import load_dotenv
load_dotenv()

from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
r = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents='Respond with only this JSON: {"status": "ok", "model": "gemma4"}',
)
print(r.text)
