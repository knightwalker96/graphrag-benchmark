"""Unified LLM client. Returns (answer_str, raw_response_dict).

Providers: "ollama" (local) or "openai" (Chat Completions API).
Only the *answering* LLM uses this; the judge in accuracy.py is independent.
"""

from __future__ import annotations
import random
import time
from typing import Any

import config

_OPENAI_MAX_RETRIES = 5
_OPENAI_BACKOFF_BASE = 2.0


def get_llm_response(prompt: str, json_mode: bool = False) -> tuple[str, Any]:
    provider = config.LLM_PROVIDER
    if provider == "ollama":
        return _ollama_call(prompt)
    if provider == "openai":
        return _openai_call(prompt, json_mode=json_mode)
    if provider == "gemini":
        return _gemini_call(prompt, json_mode=json_mode)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


_GEMINI_MAX_RETRIES = 5
_GEMINI_BACKOFF_BASE = 2.0


def _gemini_call(prompt: str, json_mode: bool = False) -> tuple[str, Any]:
    """Gemini generateContent. Returns real token usage from usageMetadata so the
    benchmark reports ACTUAL inference tokens (PS requirement), not tiktoken estimates.

    A fixed temperature (config.TEMPERATURE, default 0) keeps generation deterministic.
    Same model + params are used by all three pipelines (fairness rule)."""
    import os
    import requests

    key = config.GOOGLE_API_KEY
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set. Add it to .env for LLM_PROVIDER=gemini.")
    model = config.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    gen: dict = {"temperature": config.TEMPERATURE, "maxOutputTokens": config.MAX_TOKENS}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}

    last_err: Exception | None = None
    for attempt in range(_GEMINI_MAX_RETRIES):
        try:
            r = requests.post(url, json=body, timeout=90)
            if r.status_code == 200:
                j = r.json()
                usage = j.get("usageMetadata", {})
                cand = j.get("candidates", [{}])
                # Join ALL parts. Reading parts[0] alone silently drops the tail of any
                # multi-part response.
                content = ""
                if cand and cand[0].get("content", {}).get("parts"):
                    content = "".join(p.get("text", "") for p in cand[0]["content"]["parts"]).strip()
                # Gemini 2.5 THINKING TOKENS ARE BILLED AND COUNT AGAINST maxOutputTokens, but
                # they are reported separately from candidatesTokenCount, so a run can be
                # truncated mid-answer while `completion_tokens` looks nowhere near the cap.
                # Measured on an Olympics superlative question: candidates=11, thoughts=241,
                # cap=256 -> finishReason MAX_TOKENS and a title cut in half. Both are surfaced
                # here so the caller can record them and a truncation audit can actually see
                # truncation (an audit on completion_tokens alone cannot).
                usage_dict = {
                    "prompt_tokens":     usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens":      usage.get("totalTokenCount", 0),
                    "thinking_tokens":   usage.get("thoughtsTokenCount", 0),
                    "finish_reason":     (cand[0].get("finishReason") if cand else "") or "",
                }
                return content, {"usage": usage_dict}
            if r.status_code in (429, 500, 503):
                last_err = RuntimeError(f"Gemini {r.status_code}: {r.text[:120]}")
                time.sleep(_GEMINI_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1))
                continue
            raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
            time.sleep(_GEMINI_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1))
    raise RuntimeError(f"Gemini call failed after {_GEMINI_MAX_RETRIES} retries: {last_err}")


def _ollama_call(prompt: str) -> tuple[str, Any]:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        temperature=config.TEMPERATURE,
        num_predict=config.MAX_TOKENS,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    # Ollama doesn't return token usage — metrics.py uses tiktoken estimation
    return response.content.strip(), {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def _openai_call(prompt: str, json_mode: bool = False) -> tuple[str, Any]:
    from openai import (
        OpenAI,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
        AuthenticationError,
    )

    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )

    client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=0, timeout=60.0)

    last_err: Exception | None = None
    for attempt in range(_OPENAI_MAX_RETRIES):
        try:
            kwargs: dict = {
                "model":       config.OPENAI_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": config.TEMPERATURE,
                "max_tokens":  64 if json_mode else config.MAX_TOKENS,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            content = (response.choices[0].message.content or "").strip()
            usage = response.usage
            usage_dict = {
                "prompt_tokens":     getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens":      getattr(usage, "total_tokens", 0) if usage else 0,
            }
            return content, {"usage": usage_dict}

        except AuthenticationError as e:
            raise RuntimeError(
                f"OpenAI authentication failed ({e}). Check OPENAI_API_KEY in .env."
            ) from e

        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            last_err = e
            wait = _OPENAI_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
            kind = type(e).__name__
            print(f"[llm_client] OpenAI {kind} — retrying in {wait:.1f}s ({attempt + 1}/{_OPENAI_MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError(f"OpenAI call failed after {_OPENAI_MAX_RETRIES} retries: {last_err}")
