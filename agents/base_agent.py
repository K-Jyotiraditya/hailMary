"""
Base Agent — Foundation for all TradingGroup V2 agents.

Handles Gemini API calls, prompt templating, structured JSON parsing,
retry logic, and CoT trace logging.
"""
import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemma4:e2b"


class AgentOutput:
    """Standardized output from any agent."""
    def __init__(self, agent_name: str, data: Dict[str, Any], reasoning: str, timestamp: str = None):
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

    def build_prompt(self, context: Dict[str, Any]) -> str:
        """Build the LLM prompt from context. Override in subclass."""
        raise NotImplementedError

    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured data. Override in subclass."""
        raise NotImplementedError

    def call_llm(self, prompt: str) -> str:
        """Call local Ollama Gemma 4 with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                url = "http://localhost:11434/api/generate"
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
                response = requests.post(url, json=payload, timeout=180)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    print(f"[{self.NAME}] Local LLM call failed after {self.MAX_RETRIES} attempts: {e}")
                    return ""
                wait = 2 * (attempt + 1)
                print(f"[{self.NAME}] Retry {attempt+1}/{self.MAX_RETRIES} in {wait}s...")
                time.sleep(wait)
        return ""

    def extract_json(self, text: str) -> dict:
        """Robustly extract JSON from LLM response (handles markdown fences)."""
        text = text.strip()
        # Strip markdown code fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to find any JSON object in the text
            import re
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
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
        return AgentOutput(
            agent_name=self.NAME,
            data=data,
            reasoning=raw,
        )


def log_agent_output(output: AgentOutput, log_dir: str = "data/agent_logs"):
    """Persist agent output to daily log file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = Path(log_dir) / f"{date_str}.json"

    entries = []
    if log_path.exists():
        with open(log_path, "r") as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                entries = []

    entries.append(output.to_dict())
    with open(log_path, "w") as f:
        json.dump(entries, f, indent=2, default=str)
