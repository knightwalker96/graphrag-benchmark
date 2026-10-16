"""Central configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OPENAI_PRICING = {
    "gpt-4o-mini":     (0.00015,    0.00060),
    "gpt-4o":          (0.0025,     0.01),
    "gpt-4-turbo":     (0.01,       0.03),
    "gpt-4":           (0.03,       0.06),
    "gpt-3.5-turbo":   (0.0005,     0.0015),
}

# --- Gemini (generator; extraction/judge configured separately) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Gemini pricing per 1K tokens (input, output) — 2.5 Flash public rates.
GEMINI_PRICING = {
    "gemini-2.5-flash":      (0.00030, 0.00250),
    "gemini-2.5-flash-lite": (0.00010, 0.00040),
    "gemini-2.5-pro":        (0.00125, 0.01000),
}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

TOP_K = 10

# --- Fair-comparison context budget (reviewer fix #2) ---
# Basic RAG (P2) and GraphRAG (P3) use the SAME number of context items AND the
# SAME per-item character cap, so neither pipeline is fed more text than the other.
# 2000 chars covers the p90 of both P2 chroma chunks (~1817) and P3 docs (~2066),
# so this is a generous "level-up" cap that leaves both essentially un-starved.
CONTEXT_NUM_CHUNKS  = int(os.getenv("CONTEXT_NUM_CHUNKS", "10"))
CONTEXT_CHUNK_CHARS = int(os.getenv("CONTEXT_CHUNK_CHARS", "2000"))

MAX_DOCS_TO_INDEX = int(os.getenv("MAX_DOCS_TO_INDEX", "10000"))
EVAL_SAMPLE_SIZE = 50

# ---- Dataset selection (configurable by parameter) ----
# DATASET picks the corpus + QA source and tunes the answer style. Add new
# datasets here.
DATASET = os.getenv("DATASET", "sec")
# Max corpus tokens to load (high default = use the full available corpus).
TARGET_CORPUS_TOKENS = int(os.getenv("TARGET_CORPUS_TOKENS", "100000000"))



def ds_paths(ds: str | None = None) -> dict:
    """Standard per-dataset artifact paths (non-destructive: each dataset isolated)."""
    ds = ds or DATASET
    return {
        "chroma": f"./chroma_db_{ds}/main", "collection": f"{ds}_docs",
        "ground_truth": f"./results/ground_truth_{ds}.json",
        "index_dir": f"pipeline3_{ds}/index",
        "graph": ds,
        "benchmark": f"./results/benchmark_{ds}.json",
    }


# Chroma paths are dataset-aware — always derived from DATASET so all consumers
# (chroma_store.py, app.py) automatically point at the right index.
_ds = ds_paths()
CHROMA_PERSIST_DIR     = _ds["chroma"]
CHROMA_COLLECTION_NAME = _ds["collection"]

HF_TOKEN = os.getenv("HF_TOKEN", "")
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "hf")

BENCHMARK_RESULTS_PATH = "./results/benchmark_results.json"
GROUND_TRUTH_PATH      = "./results/ground_truth.json"

TEMPERATURE = 0.0

# Generic fallback prompt for any dataset without a tuned answer style.
_FACTOID_PROMPT = (
    "Answer the factoid question with only the exact answer — a name, date, place, "
    "number, or short noun phrase. Use the context passages if provided; otherwise "
    "use your general knowledge. No sentences. No trailing period. No 'The answer is'. "
    "No explanations."
)

# SEC filings: shared across ALL THREE pipelines verbatim (fairness rule).
# Answers are financial facts (numbers with units), names, dates, or LISTS of companies.
# Citations must reference evidence actually supplied (the [chunk_id] labels shown in
# context). LLM-only has no context, so its citation block is naturally empty.
_SEC_PROMPT = (
    "You answer questions about SEC filings (10-K annual reports, 8-K current reports, "
    "DEF 14A proxy statements) for large public companies. Use ONLY the provided context "
    "passages and knowledge-graph facts; if the answer is not supported by them, say so. "
    "Answer TERSELY — give ONLY the requested fact(s), no restated question, no preamble, "
    "no explanation:\n"
    "- Financial figures: the exact number with unit (e.g. '$416,161 million'). No extra words.\n"
    "- List/aggregation questions: output the COMPLETE set of matching companies as a "
    "comma-separated list of tickers (e.g. 'AMD, AVGO, INTC, NVDA, QCOM'); do not stop early, "
    "do not add prose.\n"
    "- 'Which had higher X: A or B' questions: name the winner and give both figures.\n"
    "- Dates and names: give them exactly, nothing more.\n"
    "After the answer, on a new line write 'Citations:' followed by the passage IDs (the "
    "[bracketed labels]) you actually used, comma-separated. Cite only passages provided."
)

# Olympics: shared across ALL THREE pipelines verbatim — the same fairness
# rule as _SEC_PROMPT, with wording that matches this corpus's answer shapes instead of SEC's.
# Measured answer shapes across the 100 public questions (OLYMPICS_DATA_NOTES.md §1):
#   lookup      -> a bare integer ("26")
#   aggregation -> a bare integer ("5")
#   superlative -> a full event title ("Shooting at the 2008 Summer Olympics – Men's 50 metre rifle prone")
#   multi_hop   -> an athlete name, sometimes a team run together with no separator
#                  ("Dani KingLaura TrottJoanna Rowsell") — the gold answer is that exact string
#   temporal    -> an athlete name
# The "copy the name exactly as written" instruction exists because of that concatenation: 268
# of 2,187 events have it, and inserting spaces would diverge from the graded reference.
#
# "ONLY the provided context" also carries real weight here that it did not on SEC. That corpus
# was synthetic and future-dated, so the model had no parametric opinion. This one is real
# Wikipedia and the generator knows the Olympics, so the corpus must be stated as the authority
# (the dataset README: "If Wikipedia today disagrees with corpus.jsonl, the corpus wins").
_OLYMPICS_PROMPT = (
    "You answer questions about Olympic events using a corpus of Wikipedia articles. The corpus "
    "is the ONLY source of truth: use ONLY the provided context passages and knowledge-graph "
    "facts, never your own knowledge of the Olympics, even if you believe the context is wrong. "
    "If the answer is not supported by the provided context, say so.\n"
    "Answer TERSELY — give ONLY the requested fact, no restated question, no preamble, no "
    "explanation:\n"
    "- 'How many ...' questions: a bare integer (e.g. '26'). No units, no words.\n"
    "- 'Who won ...' questions: the medallist's name exactly as written in the context. If the "
    "context runs several names together without spaces, reproduce that string exactly.\n"
    "- 'Which event ...' questions: the event's full title exactly as written in the context "
    "(e.g. \"Sailing at the 2000 Summer Olympics – Soling\").\n"
    "After the answer, on a new line write 'Citations:' followed by the passage IDs (the "
    "[bracketed labels]) you actually used, comma-separated. Cite only passages provided."
)

# SEC answer lists can be long (EQ44/EQ25 ~11-35 companies) — needs output headroom.
if DATASET == "sec":
    SYSTEM_PROMPT, _MAX_TOKENS_DEFAULT = _SEC_PROMPT, 512
elif DATASET == "olympics":
    # 1024, NOT the ~25 an Olympics answer needs, because GEMINI 2.5 THINKING TOKENS COUNT
    # AGAINST maxOutputTokens. Measured: a superlative question spent 241 tokens thinking and
    # had 11 left, so the event title came back as "Shooting at the 2008 Summer Olympics" with
    # finishReason MAX_TOKENS — scored as a wrong answer when the graph had served the right
    # one. `completion_tokens` read 11, nowhere near the 256 cap, so an audit on output_tokens
    # alone could not see it. Thinking is ~240-250 per call here; 1024 leaves ~770 for an
    # answer that needs ~25.
    #
    # 1024 was ALSO too tight. Thinking is far more variable than the first measurement
    # suggested: across one dev batch it ranged 0-982 tokens with a mean of 390-579 depending
    # on pipeline, and 2 rows per pipeline still hit finishReason MAX_TOKENS at 1024. 2048
    # clears the observed maximum with ~1000 to spare. The cap is NOT part of the
    # comparability contract (SEC uses 512, chosen for its 32-company list answers); what is
    # held identical is the generator, temperature and the prompt, and thinking stays ENABLED
    # here exactly as it was for SEC rather than being switched off to tidy the accounting.
    SYSTEM_PROMPT, _MAX_TOKENS_DEFAULT = _OLYMPICS_PROMPT, 2048
else:
    SYSTEM_PROMPT, _MAX_TOKENS_DEFAULT = _FACTOID_PROMPT, 32
MAX_TOKENS = int(os.getenv("MAX_TOKENS", str(_MAX_TOKENS_DEFAULT)))
