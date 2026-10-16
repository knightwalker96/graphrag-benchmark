"""Pipeline 2: Basic RAG — embed question → retrieve chunks → LLM. Primary baseline."""

from __future__ import annotations
import os
import time

from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

from metrics import CallMetrics, MetricsTimer
from llm_client import get_llm_response
import config

# Retrieval mode default: "similarity" (pure cosine top-k, no MMR, no reranker) — adopted
# per the P2 ablation study (docs/results/P2_ABLATION.md): beats mmr_rerank on strict pass,
# graded quality, AND BERTScore simultaneously, while dropping the cross-encoder dependency
# and cutting latency ~12%. All retrieval knobs are compute-only (no LLM tokens) and the
# final context stays capped at CONTEXT_NUM_CHUNKS × CONTEXT_CHUNK_CHARS — so switching modes
# keeps the P2/P3 token comparison fair.
P2_RERANK        = os.getenv("P2_RERANK", "0") != "0"
RERANK_POOL      = int(os.getenv("P2_RERANK_POOL", "40"))    # MMR candidates before rerank
MMR_LAMBDA       = float(os.getenv("P2_MMR_LAMBDA", "0.5"))  # lower = more diverse
# Retrieval mode (for ablation): "mmr_rerank" (MMR pool -> cross-encoder rerank -> top-k),
# "mmr" (MMR, no rerank), "similarity" (pure cosine top-k, no MMR, no rerank; default/best).
P2_SEARCH        = os.getenv("P2_SEARCH", "similarity")
# Ticker pre-filter: when the question names companies in the corpus, restrict retrieval to
# those companies' chunks first (fixes wrong-doc retrieval). Fair, standard metadata filtering.
P2_TICKER_FILTER = os.getenv("P2_TICKER_FILTER", "1") != "0"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_INPUT_CHARS = 600

# ---------------------------------------------------------------------------------------
# P2 context optimization (reviewer ask: "apply similar optimization/compression to vector
# RAG to isolate how much of the token saving comes specifically from graph retrieval").
# BOTH mechanisms below cost ZERO LLM tokens — adaptive-k reuses P3's deterministic regex
# router (no model call), and compression scores sentences with the SAME local embedder
# already used for retrieval. So optimizer_tokens stays 0 and the token comparison with P3
# remains apples-to-apples.
# ---------------------------------------------------------------------------------------
# Per-question k chosen by the same router P3 uses, so adaptive budgeting is not treated as
# a graph-exclusive advantage. Mirrors P3's own per-route spend.
P2_ADAPTIVE_K = os.getenv("P2_ADAPTIVE_K", "0") != "0"
# SEC route names and their frozen values. Unchanged.
_ROUTE_K_SEC = {"aggregation": 10, "lookup": 6, "bridge": 10, "compare": 10, "temporal": 10,
                "default": 10}
# Olympics has different routes (no bridge/compare; superlative and multi_hop instead). Per the
# comparability contract the SEC *values* are explicitly NOT carried over — they are re-swept on
# Olympics dev and recorded in docs/design/OLYMPICS_FROZEN_CONFIG.md. These are the pre-sweep
# placeholder (= uniform default k), so an un-swept run cannot silently inherit a SEC tuning.
_ROUTE_K_OLY = {"lookup": 10, "aggregation": 10, "superlative": 10, "multi_hop": 10,
                "temporal": 10, "default": 10}


def _route_k_table() -> dict:
    """Route -> k. `P2_ROUTE_K` ("lookup=4,aggregation=30") overrides entries so the dev sweep
    can explore profiles without editing code; the frozen profile is set here afterwards."""
    base = dict(_ROUTE_K_OLY if config.DATASET == "olympics" else _ROUTE_K_SEC)
    for part in os.getenv("P2_ROUTE_K", "").split(","):
        if "=" in part:
            kk, vv = part.split("=", 1)
            base[kk.strip()] = int(vv)
    return base


_ROUTE_K = _route_k_table()
# SEC capped adaptive-k at the global k, so it could only ever REDUCE retrieval. On Olympics that
# cap would be a thumb on the scale against the baseline: aggregation and superlative questions
# have ~11-12 gold documents each, so P2 must be able to retrieve MORE than the default 10, not
# only fewer. Default preserves the SEC behaviour exactly.
P2_ADAPTIVE_CAP = os.getenv("P2_ADAPTIVE_CAP",
                            "0" if config.DATASET == "olympics" else "1") != "0"
# Extractive sentence compression: keep only the N sentences most similar to the question,
# across the retrieved chunks. Graph-free analogue of P3's provenance-sentence substitution.
P2_COMPRESS = os.getenv("P2_COMPRESS", "0") != "0"
P2_COMPRESS_SENTS = int(os.getenv("P2_COMPRESS_SENTS", "12"))
P2_MIN_SENT_CHARS = int(os.getenv("P2_MIN_SENT_CHARS", "40"))


# Which router P2 borrows for adaptive-k, per dataset. Olympics points at the LEXICON rather
# than the pipeline: same pure function, but importing pipeline3_olympics/pipeline.py would load
# the KG index and open a TigerGraph connection inside the *baseline*, which P2 must never do.
# Pointing Olympics at the SEC router would label every question "lookup" (measured: 6/30) and
# quietly cripple adaptive-k — the exact bias this entry exists to prevent.
_ROUTER_MODULES = {
    "olympics": ("pipeline3_olympics/lexicon.py", "route_of"),
    "sec":   ("pipeline3_sec/pipeline.py", "route"),
}


def _p3_route(question: str) -> str:
    """Reuse P3's deterministic router (pure function of the question string, zero tokens)."""
    import importlib.util
    global _ROUTE_FN
    try:
        _ROUTE_FN
    except NameError:
        path, fn = _ROUTER_MODULES.get(config.DATASET, _ROUTER_MODULES["sec"])
        spec = importlib.util.spec_from_file_location("_p3r", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ROUTE_FN = getattr(mod, fn)
    return _ROUTE_FN(question)


def _compress(question: str, docs: list, max_sents: int) -> tuple[list[str], list[str]]:
    """Keep the `max_sents` sentences most similar to the question, grouped back under the
    chunk they came from (so citation IDs stay correct). Embeddings are already L2-normalized,
    so a dot product is cosine similarity."""
    import numpy as np
    from embeddings.embedder import get_embedder
    emb = get_embedder()
    sents: list[tuple[int, str]] = []
    for ci, d in enumerate(docs):
        for s in _re.split(r"(?<=[.;])\s+", d.page_content[:config.CONTEXT_CHUNK_CHARS]):
            s = s.strip()
            if len(s) >= P2_MIN_SENT_CHARS:
                sents.append((ci, s))
    if not sents:
        return ([d.page_content[:config.CONTEXT_CHUNK_CHARS] for d in docs],
                [getattr(d, "metadata", {}).get("title", "") for d in docs])
    qv = np.asarray(emb.embed_query(question))
    M = np.asarray(emb.embed_documents([s for _, s in sents]))
    order = np.argsort(-(M @ qv))[:max_sents]
    kept: dict[int, list[str]] = {}
    for idx in sorted(order, key=lambda i: (sents[i][0], i)):   # preserve reading order
        kept.setdefault(sents[idx][0], []).append(sents[idx][1])
    texts, ids = [], []
    for ci in sorted(kept):
        texts.append(" ".join(kept[ci]))
        ids.append(getattr(docs[ci], "metadata", {}).get("title", ""))
    return texts, ids                                    # model has a 512-token limit

_cross_encoder: CrossEncoder | None = None
_ticker_index: dict | None = None   # {company_name_lower / ticker_lower: TICKER}


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"[p2] Loading cross-encoder ({RERANKER_MODEL})…")
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


import re as _re

# hand aliases for companies whose common name differs from the manifest form
_TICKER_ALIASES = {
    "exxon": "XOM", "exxonmobil": "XOM", "jpmorgan": "JPM", "jp morgan": "JPM",
    "j.p. morgan": "JPM", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "facebook": "META", "berkshire": "BRK.B", "nvidia": "NVDA", "amd": "AMD",
    "home depot": "HD", "p&g": "PG", "procter": "PG", "coca-cola": "KO", "coca cola": "KO",
    "netflix": "NFLX", "adobe": "ADBE", "abbvie": "ABBV", "verizon": "VZ", "amgen": "AMGN",
    "chevron": "CVX", "colgate": "CL", "cisco": "CSCO", "caterpillar": "CAT", "costco": "COST",
    "oracle": "ORCL", "accenture": "ACN", "morgan stanley": "MS",
}


def _get_ticker_index() -> dict:
    """Map company name tokens / ticker -> TICKER (manifest + aliases), for detecting which
    companies a question is about so retrieval can pre-filter to their chunks."""
    global _ticker_index
    if _ticker_index is None:
        import json
        from pathlib import Path
        _ticker_index = dict(_TICKER_ALIASES)
        mf = Path("datasets/sec/manifest_clean.json")
        if mf.exists():
            for tk, info in json.load(open(mf)).items():
                name = (info.get("company", "") or "").lower()
                # strip corporate suffixes / parentheticals for cleaner matching
                name = _re.sub(r"\s*\(.*?\)|\s+(inc\.?|corporation|corp\.?|company|co\.?|"
                               r"plc|ltd\.?|group|holdings|&\s*co\.?)\b\.?", "", name).strip()
                _ticker_index[tk.lower()] = tk
                if name and len(name) >= 3:
                    _ticker_index.setdefault(name, tk)
    return _ticker_index


def _is_corpus_scan(question: str) -> bool:
    """Aggregation / 'scan-many-companies' questions where the answer companies are NOT named
    in the question — ticker filtering would wrongly narrow retrieval, so skip it."""
    ql = question.lower()
    return bool(_re.search(
        r"\bwhich (companies|dataset companies|.*-sector companies|.*companies)\b"
        r"|\blist (all|the)\b|\bhow many .*(compan|firm|audit)"
        r"|\bamong (the |dataset'?s )?.*companies\b"
        r"|\bwhich (energy|utilities|health|financial|consumer|industrial|communication|"
        r"materials|real estate|semiconductor|kpmg|deloitte|pwc|ernst)\b"
        r"|\bserves? on the boards?\b|\bdirector at which\b|\btwo companies\b", ql))


def _tickers_in_question(question: str) -> list[str]:
    idx = _get_ticker_index()
    ql = " " + question.lower() + " "
    found = []
    for key, tk in idx.items():
        if len(key) < 3:
            continue
        # match key as a word (allowing trailing possessive/punctuation incl. ? !)
        if _re.search(r"[\s(]" + _re.escape(key) + r"(?:'s|['.,)\s?!:;])", ql) and tk not in found:
            found.append(tk)
    return found[:4]


def build_prompt(question: str, context_chunks: list[str], chunk_ids: list[str] | None = None) -> str:
    # Label each passage with its chunk ID so the model can cite the evidence it used
    # (citation requirement). Falls back to positional labels if no IDs given.
    labels = chunk_ids or [f"Passage {i+1}" for i in range(len(context_chunks))]
    context_block = "\n\n---\n\n".join(
        f"[{labels[i] if i < len(labels) else f'Passage {i+1}'}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    )
    return (
        f"{config.SYSTEM_PROMPT}\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def _maybe_ticker_filter(question: str) -> dict | None:
    # The ticker index is built from the SEC manifest and is meaningless outside it. Measured
    # inert on Olympics (fires on 0 of all 150 dev+test+hidden questions) before this guard was
    # added, so the guard changes no Olympics number; it just stops a future question that
    # happens to contain "meta" or "oracle" from being filtered against a field Chroma has not
    # got. SEC behaviour is unchanged.
    if config.DATASET not in ("sec",):
        return None
    if P2_TICKER_FILTER and not _is_corpus_scan(question):
        tks = _tickers_in_question(question)
        if tks:
            return {"ticker": {"$in": tks}} if len(tks) > 1 else {"ticker": tks[0]}
    return None


def run_single(question: str, vectorstore: Chroma) -> CallMetrics:
    t_start = time.perf_counter()
    k = config.CONTEXT_NUM_CHUNKS
    if P2_ADAPTIVE_K:
        rk = _ROUTE_K.get(_p3_route(question), config.CONTEXT_NUM_CHUNKS)
        k = min(k, rk) if P2_ADAPTIVE_CAP else rk
    filt = _maybe_ticker_filter(question)

    if P2_SEARCH == "similarity":
        # Pure cosine similarity top-k — no MMR, no reranker.
        sk = {"k": k}
        if filt:
            sk["filter"] = filt
        selected = vectorstore.as_retriever(search_type="similarity", search_kwargs=sk).invoke(question)
        if not selected and filt:
            selected = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k}).invoke(question)
    else:
        # MMR candidate pool (diverse), optionally cross-encoder reranked.
        sk = {"k": RERANK_POOL, "fetch_k": RERANK_POOL * 3, "lambda_mult": MMR_LAMBDA}
        if filt:
            sk["filter"] = filt
        candidates = vectorstore.as_retriever(search_type="mmr", search_kwargs=sk).invoke(question)
        if not candidates and filt:
            sk.pop("filter")
            candidates = vectorstore.as_retriever(search_type="mmr", search_kwargs=sk).invoke(question)
        if P2_SEARCH == "mmr_rerank" and candidates:
            ce = _get_cross_encoder()
            scores = ce.predict([(question, d.page_content[:RERANK_INPUT_CHARS]) for d in candidates])
            candidates = [d for _, d in sorted(zip(scores, candidates), key=lambda x: -x[0])]
        selected = candidates[:k]

    if P2_COMPRESS and selected:
        context_chunks, retrieved_ids = _compress(question, selected, P2_COMPRESS_SENTS)
    else:
        context_chunks = [d.page_content[:config.CONTEXT_CHUNK_CHARS] for d in selected]
        retrieved_ids = [getattr(d, "metadata", {}).get("title", "") for d in selected]

    prompt = build_prompt(question, context_chunks, chunk_ids=retrieved_ids)
    context_text = "\n\n".join(context_chunks)
    with MetricsTimer("Pipeline 2: Basic RAG", question) as timer:
        answer, raw = get_llm_response(prompt)

    metrics = timer.record(
        answer, raw.get("usage", {}),
        question=question, context_text=context_text,
        retrieved_chunks=len(context_chunks),
        retrieved_ids=retrieved_ids,
        preserve_answer=(config.DATASET in ("sec", "olympics")),
    )
    metrics.route = "vector_rag"
    metrics.latency_seconds = round(time.perf_counter() - t_start, 3)
    return metrics


def run_batch(questions: list[str], vectorstore: Chroma, verbose: bool = False) -> list[CallMetrics]:
    from tqdm import tqdm
    results = []
    for q in tqdm(questions, desc="Pipeline 2 (Basic RAG)"):
        m = run_single(q, vectorstore)
        results.append(m)
        if verbose:
            print(f"  Q: {q[:60]}...")
            print(f"  A: {m.answer[:80]}...")
            print(f"  Chunks: {m.retrieved_chunks} | Tokens: {m.total_tokens} | Latency: {m.latency_seconds}s")
    return results
