"""
Vector Memory — ChromaDB-backed RAG store for trading decisions.

Stores each portfolio decision as an embedding so the PortfolioDecisionAgent
can retrieve semantically similar past decisions for context (reflection).

Usage:
    vm = VectorMemory()
    vm.store_decision(date="2026-05-07", weights={"NVDA": 0.105}, rationale="...", style="balanced")
    context = vm.retrieve_similar(query="tech sector momentum", n_results=3)
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


_DB_PATH = str(Path(__file__).parent / "chroma_db")
_COLLECTION = "trading_decisions"


class VectorMemory:
    def __init__(self, db_path: str = _DB_PATH):
        self._client = chromadb.PersistentClient(path=db_path)
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )

    def store_decision(
        self,
        date: str,
        weights: dict,
        rationale: str,
        style: str,
        pnl_pct: Optional[float] = None,
    ) -> None:
        """Embed and persist one portfolio decision."""
        doc = (
            f"Date: {date}. Style: {style}. "
            f"Positions: {', '.join(f'{t}={w:.1%}' for t, w in weights.items() if w > 0)}. "
            f"Rationale: {rationale}"
        )
        meta = {
            "date": date,
            "style": style,
            "weights_json": json.dumps(weights),
            "pnl_pct": pnl_pct if pnl_pct is not None else 0.0,
        }
        self._col.upsert(ids=[date], documents=[doc], metadatas=[meta])

    def retrieve_similar(self, query: str, n_results: int = 5) -> str:
        """Return a formatted string of the n most-similar past decisions."""
        count = self._col.count()
        if count == 0:
            return "No previous trading history in vector memory."

        results = self._col.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        lines = ["Recent similar decisions (RAG):"]
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            pnl = meta.get("pnl_pct", 0.0)
            pnl_str = f" | PnL: {pnl:+.2f}%" if pnl else ""
            lines.append(f"  [{meta['date']}] {meta['style'].upper()}{pnl_str}: {doc[:200]}")
        return "\n".join(lines)

    def count(self) -> int:
        return self._col.count()
