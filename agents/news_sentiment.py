"""
News-Sentiment Agent — Fetches headlines and scores with FinBERT.

Uses Finnhub for news retrieval.
Uses ProsusAI/finbert (fine-tuned BERT on financial text) for sentiment.
Falls back to Ollama LLM if FinBERT is unavailable.

FinBERT is dramatically more accurate than a general-purpose LLM for
financial sentiment: it was trained on financial news, analyst reports,
and earnings call transcripts.
"""
import os
import numpy as np
import finnhub
from datetime import datetime, timedelta
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
from agents.base_agent import BaseAgent, AgentOutput

_fh_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY", ""))

# Lazy-load FinBERT once, shared across all agent instances
_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        print("  [FinBERT] Loading ProsusAI/finbert (first run downloads ~440MB)...")
        _finbert_pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,          # CPU
            truncation=True,
            max_length=512,
            top_k=None,         # return all 3 label scores
        )
        print("  [FinBERT] Model loaded.")
        return _finbert_pipeline
    except Exception as e:
        print(f"  [FinBERT] Load failed: {e} — falling back to LLM")
        return None


def _score_with_finbert(headlines: list) -> dict:
    """
    Score a list of headline strings with FinBERT.
    Returns {sentiment_score, confidence, key_theme}.
    """
    pipe = _get_finbert()
    if pipe is None or not headlines:
        return None

    label_to_score = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = []
    confidences = []

    for text in headlines[:8]:  # cap at 8 headlines
        try:
            results = pipe(text[:512])
            # results is list of [{"label": ..., "score": ...}, ...]
            if isinstance(results[0], list):
                results = results[0]
            best = max(results, key=lambda x: x["score"])
            num_score = label_to_score.get(best["label"].lower(), 0.0)
            scores.append(num_score * best["score"])
            confidences.append(best["score"])
        except Exception:
            continue

    if not scores:
        return None

    # Weighted average (more confident predictions count more)
    weights = np.array(confidences)
    weighted_score = float(np.average(scores, weights=weights))
    avg_confidence = float(np.mean(confidences))

    # Dominant theme from the most confident headline
    dominant_idx = int(np.argmax(confidences))
    key_theme = headlines[dominant_idx][:100] if dominant_idx < len(headlines) else ""

    return {
        "sentiment_score": round(np.clip(weighted_score, -1.0, 1.0), 3),
        "confidence":      round(avg_confidence, 3),
        "key_theme":       key_theme,
        "source":          "finbert",
    }


class NewsSentimentAgent(BaseAgent):
    NAME = "News-Sentiment"

    def _fetch_news(self, ticker: str, days_back: int = 3) -> list:
        end   = datetime.now()
        start = end - timedelta(days=days_back)
        try:
            news = _fh_client.company_news(
                ticker,
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
            )
            seen, filtered = set(), []
            for article in news[:15]:
                h = article.get("headline", "")
                if h and h not in seen:
                    seen.add(h)
                    filtered.append({
                        "headline": h,
                        "summary":  article.get("summary", "")[:200],
                        "source":   article.get("source", ""),
                    })
                if len(filtered) >= 8:
                    break
            return filtered
        except Exception as e:
            print(f"[News] Finnhub error for {ticker}: {e}")
            return []

    def build_prompt(self, context: Dict[str, Any]) -> str:
        ticker = context["ticker"]
        news   = context["news"]
        news_block = "\n".join(
            f"  {i+1}. [{n['source']}] {n['headline']}\n     {n['summary']}"
            for i, n in enumerate(news)
        ) or "  No recent news available."

        return f"""You are a financial news analyst. Analyze the following recent news for {ticker}.

RECENT NEWS FOR {ticker}:
{news_block}

Respond with ONLY a JSON object:
{{
  "sentiment_score": <float -1.0 to +1.0>,
  "confidence": <float 0.0-1.0>,
  "key_theme": "<one-line dominant narrative>"
}}"""

    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        data = self.extract_json(raw_text)
        return {
            "sentiment_score": float(data.get("sentiment_score", 0.0)),
            "confidence":      float(data.get("confidence", 0.5)),
            "key_theme":       data.get("key_theme", ""),
            "source":          "llm",
        }

    def run(self, context: Dict[str, Any]) -> AgentOutput:
        ticker = context["ticker"]
        news   = self._fetch_news(ticker)

        if not news:
            return AgentOutput(
                agent_name=self.NAME,
                data={"sentiment_score": 0.0, "confidence": 0.0,
                      "key_theme": "No news available", "source": "none"},
                reasoning="No recent news found.",
            )

        headlines = [n["headline"] for n in news]

        # ── Try FinBERT first ─────────────────────────────────────────────────
        finbert_result = _score_with_finbert(headlines)
        if finbert_result is not None:
            return AgentOutput(
                agent_name=self.NAME,
                data=finbert_result,
                reasoning=f"FinBERT scored {len(headlines)} headlines",
            )

        # ── Fallback: LLM ─────────────────────────────────────────────────────
        context["news"] = news
        return super().run(context)
