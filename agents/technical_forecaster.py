"""
Technical-Forecaster Agent — Combines technical indicators with LLM reasoning.

Computes RSI-14, 20-day SMA distance, ATR-20, HV-10 (matching the paper),
then feeds them to Gemini with a hybrid gate that hard-intercepts
overbought conditions without valid breakout.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from agents.base_agent import BaseAgent
from data.market_data import yf_download


class TechnicalForecasterAgent(BaseAgent):
    NAME = "Technical-Forecaster"

    def _compute_indicators(self, ticker: str) -> dict:
        """Compute all technical indicators matching the TradingGroup paper."""
        try:
            data = yf_download(ticker, period="60d", progress=False, multi_level_index=False)
            if data.empty or len(data) < 21:
                return {}

            close = data["Close"]
            log_ret = np.log(close / close.shift(1)).dropna()

            # RSI-14
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            rs = avg_gain / avg_loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]

            # Distance to 20-day SMA (%)
            sma20 = close.rolling(20).mean().iloc[-1]
            sma_dist = ((close.iloc[-1] - sma20) / sma20) * 100

            # Distance to 20-day High/Low (%)
            high20 = close.rolling(20).max().iloc[-1]
            low20 = close.rolling(20).min().iloc[-1]
            dist_high = ((close.iloc[-1] - high20) / high20) * 100
            dist_low = ((close.iloc[-1] - low20) / low20) * 100

            # HV-10 (annualized)
            hv10 = float(log_ret.tail(10).std() * np.sqrt(252) * 100)

            # Simplified ATR-20 (std of 20d log returns)
            atr20 = float(log_ret.tail(20).std() * 100)

            # Breakout threshold (paper formula)
            breakout_thresh = max(1.0, 0.5 * atr20)

            # 20-day High/Low flag
            new_high = close.iloc[-1] >= high20
            new_low = close.iloc[-1] <= low20

            return {
                "price": round(float(close.iloc[-1]), 2),
                "rsi_14": round(float(rsi), 1),
                "sma20_dist_pct": round(float(sma_dist), 2),
                "dist_to_20d_high_pct": round(float(dist_high), 2),
                "dist_to_20d_low_pct": round(float(dist_low), 2),
                "hv_10_annualized": round(hv10, 1),
                "atr_20_pct": round(atr20, 2),
                "breakout_threshold": round(breakout_thresh, 2),
                "new_20d_high": bool(new_high),
                "new_20d_low": bool(new_low),
            }
        except Exception as e:
            print(f"[Technical] Error computing indicators for {ticker}: {e}")
            return {}

    def _hybrid_gate(self, indicators: dict, llm_output: dict) -> dict:
        """
        Hybrid Gate from the paper:
        - Hard intercept: RSI overbought + no valid breakout → force sideways
        - Soft pass: LLM uptrend prob > threshold + bullish technical → uptrend
        """
        rsi = indicators.get("rsi_14", 50)
        dist_high = abs(indicators.get("dist_to_20d_high_pct", 0))
        breakout_thresh = indicators.get("breakout_threshold", 1.0)
        new_high = indicators.get("new_20d_high", False)

        direction = llm_output.get("direction", "sideways")
        confidence = llm_output.get("confidence", 0.5)

        # Hard intercept: overbought and NOT breaking out
        if rsi > 70 and dist_high > breakout_thresh and not new_high:
            return {"direction": "sideways", "confidence": 0.3,
                    "gate": "HARD_INTERCEPT: RSI overbought, no valid breakout"}

        # Soft pass: keep LLM output
        return {"direction": direction, "confidence": confidence, "gate": "SOFT_PASS"}

    def build_prompt(self, context: Dict[str, Any]) -> str:
        ticker = context["ticker"]
        ind = context["indicators"]
        sentiment = context.get("sentiment_score", 0.0)
        news_theme = context.get("news_theme", "N/A")

        return f"""You are a quantitative stock analyst. Given the technical indicators and news sentiment for {ticker}, predict the most likely price direction for the NEXT trading day.

TECHNICAL INDICATORS FOR {ticker}:
  Price: ${ind.get('price', 'N/A')}
  RSI-14: {ind.get('rsi_14', 'N/A')}
  Distance to 20-day SMA: {ind.get('sma20_dist_pct', 'N/A')}%
  Distance to 20-day High: {ind.get('dist_to_20d_high_pct', 'N/A')}%
  Distance to 20-day Low: {ind.get('dist_to_20d_low_pct', 'N/A')}%
  10-day Historical Volatility (annualized): {ind.get('hv_10_annualized', 'N/A')}%
  20-day ATR: {ind.get('atr_20_pct', 'N/A')}%
  New 20-day High: {ind.get('new_20d_high', False)}
  New 20-day Low: {ind.get('new_20d_low', False)}

NEWS SENTIMENT: {sentiment:.2f} (-1=bearish, +1=bullish)
NEWS THEME: {news_theme}

Respond with ONLY a JSON object:
{{
  "direction": "<up|down|sideways>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one line explanation>"
}}"""

    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        data = self.extract_json(raw_text)
        return {
            "direction": data.get("direction", "sideways"),
            "confidence": float(data.get("confidence", 0.5)),
            "reasoning": data.get("reasoning", ""),
        }

    def run(self, context: Dict[str, Any]):
        from agents.base_agent import AgentOutput

        ticker = context["ticker"]

        # ── Try XGBoost model first ───────────────────────────────────────────
        try:
            from data.feature_engineer import get_live_features
            from models.xgboost_trainer import predict as xgb_predict, MODEL_PATH
            if MODEL_PATH.exists():
                feats = get_live_features(ticker)
                if feats:
                    prob, direction = xgb_predict(feats)
                    confidence = abs(prob - 0.5) * 2  # map 0.5→0, 1.0→1
                    indicators = self._compute_indicators(ticker)
                    gate_result = self._hybrid_gate(indicators or {}, {"direction": direction, "confidence": confidence})
                    return AgentOutput(self.NAME, {
                        "direction":  gate_result["direction"],
                        "confidence": round(gate_result["confidence"], 3),
                        "gate":       gate_result["gate"],
                        "prob_up":    round(prob, 3),
                        "source":     "xgboost",
                        "indicators": indicators or {},
                    }, f"XGBoost p_up={prob:.3f}")
        except Exception as e:
            print(f"[Technical] XGBoost inference failed for {ticker}: {e} — falling back to LLM")

        # ── Fallback: LLM ────────────────────────────────────────────────────
        indicators = self._compute_indicators(ticker)
        if not indicators:
            return AgentOutput(self.NAME, {"direction": "sideways", "confidence": 0.0, "gate": "NO_DATA"}, "No data")

        context["indicators"] = indicators
        prompt = self.build_prompt(context)
        raw = self.call_llm(prompt)
        llm_output = self.parse_response(raw)

        gated = self._hybrid_gate(indicators, llm_output)
        gated["indicators"] = indicators
        gated["source"] = "llm"
        gated["reasoning"] = llm_output.get("reasoning", "") + f" | Gate: {gated['gate']}"

        return AgentOutput(self.NAME, gated, raw)
