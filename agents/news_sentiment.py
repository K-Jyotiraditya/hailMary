"""
News-Sentiment Agent — Fetches and analyzes financial news for each stock.

Uses Finnhub's free API for real-time news retrieval, then prompts
Gemini to score overall sentiment from -1.0 (bearish) to +1.0 (bullish).
"""
import os
import finnhub
from datetime import datetime, timedelta
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
from agents.base_agent import BaseAgent

_fh_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))


class NewsSentimentAgent(BaseAgent):
    NAME = "News-Sentiment"

    def _fetch_news(self, ticker: str, days_back: int = 3) -> list:
        """Fetch recent news from Finnhub."""
        end = datetime.now()
        start = end - timedelta(days=days_back)
        try:
            news = _fh_client.company_news(
                ticker,
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
            )
            # Take top 5 most recent, deduplicate by headline
            seen = set()
            filtered = []
            for article in news[:15]:
                h = article.get("headline", "")
                if h and h not in seen:
                    seen.add(h)
                    filtered.append({
                        "headline": h,
                        "summary": article.get("summary", "")[:200],
                        "source": article.get("source", ""),
                    })
                if len(filtered) >= 5:
                    break
            return filtered
        except Exception as e:
            print(f"[News] Finnhub error for {ticker}: {e}")
            return []

    def build_prompt(self, context: Dict[str, Any]) -> str:
        ticker = context["ticker"]
        news = context["news"]

        news_block = "\n".join(
            f"  {i+1}. [{n['source']}] {n['headline']}\n     {n['summary']}"
            for i, n in enumerate(news)
        )
        if not news_block:
            news_block = "  No recent news available."

        return f"""You are a financial news analyst. Analyze the following recent news for {ticker} and determine the overall market sentiment.

RECENT NEWS FOR {ticker}:
{news_block}

Respond with ONLY a JSON object (no markdown, no explanation outside the JSON):
{{
  "sentiment_score": <float from -1.0 (very bearish) to +1.0 (very bullish)>,
  "confidence": <float from 0.0 to 1.0>,
  "key_theme": "<one-line summary of the dominant narrative>"
}}"""

    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        data = self.extract_json(raw_text)
        return {
            "sentiment_score": float(data.get("sentiment_score", 0.0)),
            "confidence": float(data.get("confidence", 0.5)),
            "key_theme": data.get("key_theme", "No analysis available"),
        }

    def run(self, context: Dict[str, Any]):
        ticker = context["ticker"]
        news = self._fetch_news(ticker)
        context["news"] = news
        
        if not news:
            from agents.base_agent import AgentOutput
            return AgentOutput(
                agent_name=self.NAME,
                data={"sentiment_score": 0.0, "confidence": 0.0, "key_theme": "No news available"},
                reasoning="No recent news found for this ticker.",
            )
        
        return super().run(context)
