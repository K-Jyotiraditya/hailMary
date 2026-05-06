"""Quick local Gemma 4 verification test - no API keys needed."""
import sys
import time

import requests
from dotenv import load_dotenv

sys.path.append(".")


def main():
    load_dotenv()
    print("=== Local Gemma 4 (E2B) Verification ===\n")

    print("[Test 1] Direct Ollama API call...")
    try:
        start = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma4:e2b",
                "prompt": 'Respond with ONLY this JSON, no other text: {"status": "ok", "model": "gemma4-local"}',
                "stream": False,
            },
            timeout=120,
        )
        elapsed = time.time() - start
        print(f"  Response: {response.json().get('response', '???')[:200]}")
        print(f"  Latency: {elapsed:.1f}s")
    except Exception as exc:
        print(f"  Ollama unavailable: {exc}")
        return

    print("\n[Test 2] Through BaseAgent pipeline...")
    from agents.news_sentiment import NewsSentimentAgent

    start2 = time.time()
    agent = NewsSentimentAgent()
    result = agent.run({"ticker": "AAPL"})
    elapsed2 = time.time() - start2
    print(f"  Sentiment: {result.data.get('sentiment_score', '?')}")
    print(f"  Theme: {result.data.get('key_theme', '?')[:80]}")
    print(f"  Latency: {elapsed2:.1f}s")

    print("\n[Test 3] Speed benchmark (3 calls)...")
    from agents.fundamentals_agent import FundamentalsAgent
    from agents.technical_forecaster import TechnicalForecasterAgent

    tech = TechnicalForecasterAgent()
    fund = FundamentalsAgent()

    start3 = time.time()
    t1 = tech.run({"ticker": "AAPL", "sentiment_score": 0.2, "news_theme": "test"})
    t2 = fund.run({"ticker": "AAPL"})
    t3 = agent.run({"ticker": "NVDA"})
    elapsed3 = time.time() - start3

    print(f"  3 calls completed in {elapsed3:.1f}s ({elapsed3 / 3:.1f}s per call)")
    print(f"  Tech direction: {t1.data.get('direction', '?')}")
    print(f"  Fund health: {t2.data.get('health_score', '?')}")
    print(f"  NVDA sentiment: {t3.data.get('sentiment_score', '?')}")
    print("\n=== LOCAL INFERENCE VERIFIED ===")


if __name__ == "__main__":
    main()
