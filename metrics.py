"""Token counting, latency, and cost tracking per pipeline call."""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, asdict, field

import tiktoken

import config


_PREFIX_RE = re.compile(
    r'^(the\s+answer\s+is[:,]?\s*'
    r'|answer[:,]?\s*'
    r'|based\s+on\s+(the\s+)?(context|passages|information)[:,]?\s*'
    r'|according\s+to\s+(the\s+)?(context|passage|information)[:,]?\s*)',
    re.IGNORECASE,
)


def normalize_answer(s: str) -> str:
    """Strip preambles, trailing punctuation, quotes - keeps short factoid answers clean."""
    if not s:
        return s
    s = s.strip()
    s = s.strip('"""\'`')
    s = _PREFIX_RE.sub('', s)
    # Take only the first non-empty line - guards against models appending
    # explanation after "Answer: <factoid>\nExplanation: ..."
    first_line = s.split('\n')[0].strip()
    if first_line:
        s = first_line
    s = s.rstrip('.,;:! \t\n')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


@dataclass
class CallMetrics:
    pipeline: str = ""
    question: str = ""
    answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    retrieved_chunks: int = 0

    # --- Token accounting ---
    # Actual API usage from the generator whenever available; the breakdown lets the
    # dashboard show exactly where tokens go. token_source flags actual vs estimated.
    token_source: str = "estimated"     # "actual" (from API usage) | "estimated" (tiktoken)
    # Gemini 2.5 reasoning tokens. BILLED as output and counted against maxOutputTokens, but
    # reported separately from candidatesTokenCount — so they are invisible in
    # total_inference_tokens, and a truncation audit based on output_tokens cannot see a
    # MAX_TOKENS cut-off caused by them. Recorded so both facts are checkable.
    thinking_tokens: int = 0
    finish_reason: str = ""             # "STOP" | "MAX_TOKENS" | ... — "" when the provider
                                        # does not report one (OpenAI/Ollama paths)
    system_tokens: int = 0              # system + instruction prompt tokens
    question_tokens: int = 0           # the question itself
    context_tokens: int = 0            # retrieved evidence placed in the prompt
    optimizer_tokens: int = 0          # tokens spent on router/rerank/filter/compress LLM calls
    output_tokens: int = 0             # generated answer tokens (== completion_tokens for the answer call)
    total_inference_tokens: int = 0    # across ALL model calls needed to answer (answer + optimizer)
    num_llm_calls: int = 1             # how many generator/optimizer calls this question took
    citations: list = field(default_factory=list)  # citation IDs parsed from the answer

    # --- Pipeline-3 graph provenance (reviewer fix #1c) ---
    # Proof, per question, that the LIVE cloud graph was queried and what it returned.
    # Defaults keep P1/P2 rows clean (no graph => tg_live False, empty doc list).
    tg_live: bool = False               # True only if >=1 live TigerGraph query ran FOR THIS QUESTION
                                        # (derived from tg_queries; previously a session-level flag
                                        #  that reported True even for questions answered entirely
                                        #  from the local index — see tg_queries below)
    tg_queries: int = 0                 # live GSQL queries actually issued for this question.
                                        # 0 is legitimate: several aggregation paths (has_director/
                                        # has_ceo surname match, __eventf__/__section__ scans,
                                        # corroborated competitor_of) answer from the local index
                                        # or corpus and never touch Savanna.
    tg_host: str = ""                   # which Savanna instance answered
    graph_doc_ids: list = field(default_factory=list)  # doc IDs the graph BFS returned
    graph_entity_count: int = 0         # entities the graph expansion surfaced
    route: str = ""                     # question route chosen by P3 (lookup/bridge/aggregation/temporal)
    # Which installed graph queries this question issued, IN ORDER. Added in the Olympics corpus because the
    # SEC round shipped a two-hop `bridge` branch that scored 1/20 by resolving hop 1 and then
    # never issuing hop 2 (the SEC study_EXTRA100_RESULTS §5.1) — a defect that a hop log makes visible
    # on the first dev run instead of after a paid held-out batch. Empty for P1/P2 and for the
    # SEC pipeline, which does not populate it, so no existing row's metrics change.
    hop_log: list = field(default_factory=list)
    # True if a graph BRANCH answered; False if the question fell through to vector retrieval.
    # Distinguishes "the graph had no answer" from "the router sent it somewhere with no branch".
    branch_resolved: bool = False
    retrieved_ids: list = field(default_factory=list)  # chunk/doc IDs supplied to the LLM

    def to_dict(self) -> dict:
        return asdict(self)


def count_tokens_tiktoken(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    if config.LLM_PROVIDER == "openai":
        rates = config.OPENAI_PRICING.get(config.OPENAI_MODEL)
    elif config.LLM_PROVIDER == "gemini":
        rates = config.GEMINI_PRICING.get(config.GEMINI_MODEL)
    else:
        rates = None
    if not rates:
        return 0.0
    in_per_1k, out_per_1k = rates
    return (prompt_tokens / 1000.0) * in_per_1k + (completion_tokens / 1000.0) * out_per_1k


class MetricsTimer:
    def __init__(self, pipeline_name: str, question: str):
        self.pipeline = pipeline_name
        self.question = question
        self._start: float = 0.0
        self.metrics = CallMetrics(pipeline=pipeline_name, question=question)

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.metrics.latency_seconds = round(time.perf_counter() - self._start, 3)

    def record_manual(self, prompt: str, answer: str, retrieved_chunks: int = 0) -> CallMetrics:
        cleaned = normalize_answer(answer)
        self.metrics.answer = cleaned
        self.metrics.prompt_tokens = count_tokens_tiktoken(prompt)
        self.metrics.completion_tokens = count_tokens_tiktoken(cleaned)
        self.metrics.total_tokens = self.metrics.prompt_tokens + self.metrics.completion_tokens
        self.metrics.cost_usd = calculate_cost(self.metrics.prompt_tokens, self.metrics.completion_tokens)
        self.metrics.retrieved_chunks = retrieved_chunks
        return self.metrics

    def record(
        self,
        answer: str,
        usage: dict,
        *,
        question: str = "",
        context_text: str = "",
        retrieved_chunks: int = 0,
        retrieved_ids: list | None = None,
        optimizer_tokens: int = 0,
        num_llm_calls: int = 1,
        preserve_answer: bool = False,
    ) -> CallMetrics:
        """Record a call using ACTUAL API usage (usage={'prompt_tokens','completion_tokens',
        'total_tokens'}). Falls back to tiktoken only if usage is empty. Fills the PS
        token breakdown: system, question, context, optimizer, output, total-inference.

        preserve_answer keeps list answers intact (normalize_answer truncates to the first
        line, which would clip multi-company list answers)."""
        cleaned = answer.strip() if preserve_answer else normalize_answer(answer)
        m = self.metrics
        m.answer = cleaned

        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        if pt or ct:
            m.token_source = "actual"
        else:  # provider returned no usage — estimate
            m.token_source = "estimated"
            pt = count_tokens_tiktoken(context_text) + count_tokens_tiktoken(question) + 40
            ct = count_tokens_tiktoken(cleaned)

        m.thinking_tokens = int(usage.get("thinking_tokens", 0) or 0)
        m.finish_reason = str(usage.get("finish_reason", "") or "")
        m.prompt_tokens = pt
        m.completion_tokens = ct
        m.output_tokens = ct
        m.total_tokens = pt + ct

        # Breakdown (estimated split of the prompt into its parts — the API only gives a
        # single prompt total, so we attribute proportionally by tiktoken sub-counts).
        q_tok = count_tokens_tiktoken(question) if question else 0
        ctx_tok = count_tokens_tiktoken(context_text) if context_text else 0
        sys_tok = count_tokens_tiktoken(config.SYSTEM_PROMPT)
        m.question_tokens = q_tok
        m.context_tokens = ctx_tok
        m.system_tokens = sys_tok
        m.optimizer_tokens = optimizer_tokens
        m.num_llm_calls = num_llm_calls
        # total inference = this answer call's total + any optimizer/router/rerank LLM tokens
        m.total_inference_tokens = m.total_tokens + optimizer_tokens

        m.retrieved_chunks = retrieved_chunks
        m.retrieved_ids = retrieved_ids or []
        m.citations = parse_citations(answer)
        m.cost_usd = calculate_cost(pt + _optimizer_prompt_share(optimizer_tokens),
                                    ct + _optimizer_output_share(optimizer_tokens))
        return m


def _optimizer_prompt_share(optimizer_tokens: int) -> int:
    # optimizer tokens lump prompt+output; assume ~80/20 for costing (input-heavy).
    return int(optimizer_tokens * 0.8)


def _optimizer_output_share(optimizer_tokens: int) -> int:
    return int(optimizer_tokens * 0.2)


_CITE_RE = re.compile(r"\[([^\]\n]{2,120})\]")


def parse_citations(answer: str) -> list[str]:
    """Extract citation IDs from an answer. Supports a trailing 'Citations: a, b, c'
    line and inline [bracketed] IDs."""
    cites: list[str] = []
    m = re.search(r"citations?\s*:\s*(.+)$", answer, re.IGNORECASE | re.DOTALL)
    if m:
        tail = m.group(1)
        for part in re.split(r"[,\n]", tail):
            p = part.strip().strip("[]").strip()
            if p and p.lower() not in ("none", "n/a"):
                cites.append(p)
    if not cites:
        cites = [c.strip() for c in _CITE_RE.findall(answer)]
    # dedupe, keep order
    seen, out = set(), []
    for c in cites:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def aggregate_metrics(all_metrics: list[CallMetrics]) -> dict:
    if not all_metrics:
        return {}
    n = len(all_metrics)
    return {
        "pipeline": all_metrics[0].pipeline,
        "num_queries": n,
        "avg_prompt_tokens": round(sum(m.prompt_tokens for m in all_metrics) / n, 1),
        "avg_completion_tokens": round(sum(m.completion_tokens for m in all_metrics) / n, 1),
        "avg_total_tokens": round(sum(m.total_tokens for m in all_metrics) / n, 1),
        # --- PS token accounting rollup ---
        "avg_context_tokens": round(sum(m.context_tokens for m in all_metrics) / n, 1),
        "avg_optimizer_tokens": round(sum(m.optimizer_tokens for m in all_metrics) / n, 1),
        "avg_output_tokens": round(sum(m.output_tokens for m in all_metrics) / n, 1),
        "avg_total_inference_tokens": round(sum(m.total_inference_tokens for m in all_metrics) / n, 1),
        "total_inference_tokens": sum(m.total_inference_tokens for m in all_metrics),
        "avg_num_llm_calls": round(sum(m.num_llm_calls for m in all_metrics) / n, 2),
        "token_source": all_metrics[0].token_source,
        "avg_latency_seconds": round(sum(m.latency_seconds for m in all_metrics) / n, 3),
        "avg_cost_usd": round(sum(m.cost_usd for m in all_metrics) / n, 6),
        "total_cost_usd": round(sum(m.cost_usd for m in all_metrics), 4),
        "avg_retrieved_chunks": round(sum(m.retrieved_chunks for m in all_metrics) / n, 1),
        "avg_citations": round(sum(len(m.citations) for m in all_metrics) / n, 1),
        # --- Graph provenance rollup (reviewer fix #1c) ---
        # tg_live_rate = fraction of questions the live cloud graph actually answered.
        # This is NOT expected to be 1.0. Pipeline 3 is deliberately hybrid: routes that are
        # answered from the local provenance index (edges_prov.json) or from chunk metadata issue
        # no GSQL query and so report tg_live=False. The final system combined-150 batch sits at 0.947.
        # A rate of 0.0 means the graph is unreachable and the run is invalid; anything above that
        # is a measurement of how often the live graph was the chosen path, not a pass/fail gate.
        # avg_graph_docs shows the graph's reach. See V7_OPTIMIZATION_STUDY.md §6.
        "tg_live_rate": round(sum(1 for m in all_metrics if m.tg_live) / n, 3),
        "avg_graph_docs": round(sum(len(m.graph_doc_ids) for m in all_metrics) / n, 1),
    }
