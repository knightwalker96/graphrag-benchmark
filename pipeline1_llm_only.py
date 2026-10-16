"""Pipeline 1: Raw LLM — no retrieval. Baseline to beat."""

from __future__ import annotations

from metrics import CallMetrics, MetricsTimer
from llm_client import get_llm_response
import config


def build_prompt(question: str) -> str:
    return f"{config.SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"


def run_single(question: str) -> CallMetrics:
    prompt = build_prompt(question)
    with MetricsTimer("Pipeline 1: LLM-Only", question) as timer:
        answer, raw = get_llm_response(prompt)
    metrics = timer.record(
        answer, raw.get("usage", {}),
        question=question, context_text="", retrieved_chunks=0,
        preserve_answer=(config.DATASET in ("sec", "olympics")),
    )
    metrics.route = "llm_only"
    return metrics


def run_batch(questions: list[str], verbose: bool = False) -> list[CallMetrics]:
    from tqdm import tqdm
    results = []
    for q in tqdm(questions, desc="Pipeline 1 (LLM-Only)"):
        m = run_single(q)
        results.append(m)
        if verbose:
            print(f"  Q: {q[:60]}...")
            print(f"  A: {m.answer[:80]}...")
            print(f"  Tokens: {m.total_tokens} | Latency: {m.latency_seconds}s | Cost: ${m.cost_usd:.6f}")
    return results
