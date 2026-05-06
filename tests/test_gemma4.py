"""Quick Gemma 4 validation test."""
from dotenv import load_dotenv
import os
from google import genai


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Skipping remote Gemma smoke test.")
        return

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemma-4-26b-a4b-it",
            contents='Respond with only this JSON: {"status": "ok", "model": "gemma4"}',
        )
        print(response.text)
    except Exception as exc:
        print(f"Remote Gemma smoke test unavailable: {exc}")


if __name__ == "__main__":
    main()
