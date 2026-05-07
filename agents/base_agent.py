"""
Base Agent — Foundation for all TradingGroup V2 agents.

Handles Ollama LLM calls, prompt templating, structured JSON parsing,
retry logic, and agent output logging.
"""
import json
import math
import os
import re
import threading
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Limit concurrent Ollama requests — local models choke when all 5 workers
# fire simultaneously. 3 concurrent calls saturates a typical local GPU/CPU
# without triggering 500 errors.
_OLLAMA_SEM = threading.Semaphore(3)


class AgentOutput:
    """Standardized output from any agent."""
    def __init__(self, agent_name: str, data: Dict[str, Any], reasoning: str,
                 timestamp: Optional[str] = None):
        self.agent_name = agent_name
        self.data = data
        self.reasoning = reasoning
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "data": self.data,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }


class BaseAgent:
    """
    Abstract base for all trading agents.
    Subclasses implement build_prompt() and parse_response().
    """
    NAME = "BaseAgent"
    MAX_RETRIES = 3

    def __init__(self):
        self.model = MODEL
        self.ollama_url = OLLAMA_URL
        self.session = requests.Session()

    def build_prompt(self, context: Dict[str, Any]) -> str:
        raise NotImplementedError

    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        raise NotImplementedError

    def call_llm(self, prompt: str) -> str:
        """Call local Ollama Gemma 4 with retry logic and concurrency throttle."""
        for attempt in range(self.MAX_RETRIES):
            try:
                with _OLLAMA_SEM:
                    response = self.session.post(
                        self.ollama_url,
                        json={"model": self.model, "prompt": prompt, "stream": False},
                        timeout=180,
                    )
                response.raise_for_status()
                body = response.json()
                raw = body.get("response", "") if isinstance(body, dict) else ""
                if not raw.strip():
                    print(f"[{self.NAME}] WARNING: LLM returned empty response — using fallback defaults")
                return raw
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    print(f"[{self.NAME}] LLM call failed after {self.MAX_RETRIES} attempts: {e}")
                    return ""
                wait = 2 * (attempt + 1)
                print(f"[{self.NAME}] Retry {attempt + 1}/{self.MAX_RETRIES} in {wait}s...")
                time.sleep(wait)
        return ""

    def extract_json(self, text: str) -> dict:
        """Robustly extract a JSON object from LLM response text.

        Handles markdown code fences, leading/trailing prose, and nested
        JSON objects (e.g. portfolio weights dict inside the outer object).
        """
        text = text.strip()

        # Strip markdown fences first
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Try the whole stripped text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Fallback: find the outermost {...} block (supports nested objects)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}

    def run(self, context: Dict[str, Any]) -> AgentOutput:
        """Execute the agent pipeline: build prompt → call LLM → parse."""
        prompt = self.build_prompt(context)
        raw = self.call_llm(prompt)
        data = self.parse_response(raw)
        return AgentOutput(agent_name=self.NAME, data=data, reasoning=raw)


def _sanitize(obj):
    """Recursively replace NaN/inf floats with None so json.dump produces valid JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


_LOG_LOCK = threading.Lock()


def log_agent_output(output: AgentOutput, log_dir: str = "data/agent_logs", date_str: Optional[str] = None):
    """Persist agent output to the daily log file (thread-safe)."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    day = date_str or datetime.now().strftime('%Y-%m-%d')
    log_path = Path(log_dir) / f"{day}.json"

    with _LOG_LOCK:
        entries: list = []
        if log_path.exists():
            try:
                with open(log_path) as f:
                    entries = json.load(f)
            except json.JSONDecodeError:
                entries = []

        entries.append(output.to_dict())
        with open(log_path, "w") as f:
            json.dump(_sanitize(entries), f, indent=2, default=str)
