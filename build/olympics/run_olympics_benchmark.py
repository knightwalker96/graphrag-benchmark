"""Thin wrapper over the shared benchmark harness, for the Olympics splits.

The harness (`build/bench/run_benchmark.py`) does all the work. This file exists only
to (a) pin the environment that the comparability contract fixes, and (b) spell the two
Olympics-specific arguments so they cannot be forgotten on the command line.

Three datasets share one harness. If this file ever grows past ~60 lines of logic, the harness
is not generic enough and the harness should be fixed instead of forked — that genericity is
part of what the Olympics corpus is testing.

Two harness changes were needed, both additive and both defaulting to the SEC behaviour so
every the SEC corpus reproduction command is unaffected:
  * SPLITS gained "olympics_dev" / "olympics_test";
  * run_pipeline() gained `p3_module`, defaulting to pipeline3_sec/pipeline.py.
It also now carries `gold_doc_ids` and `qtype` onto each row (empty on SEC splits) so retrieval
precision/recall@k is scorable.

Run:
    python "build/olympics/run_olympics_benchmark.py" --split dev  --runs 1 --pipelines 1,2
    python "build/olympics/run_olympics_benchmark.py" --split test --runs 3 --pipelines 1,2,3

WHY THE ENVIRONMENT IS SET HERE AND NOT LEFT TO THE SHELL
    zsh does not word-split `env $VAR`, and two arms of an earlier round were silently run with
    the wrong settings that way. EMBEDDING_MODEL in particular MUST be set: the default is
    all-mpnet-base-v2 (768-dim) and the Olympics Chroma collection holds 384-dim vectors, so
    the mistake surfaces as a dimension error at best and a wrong collection at worst.
    Anything already exported in the shell wins, so ablation arms can still override.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "build" / "run_benchmark.py"
P3_MODULE = "pipeline3_olympics/pipeline.py"

# The frozen comparability contract. setdefault, never assignment, so an explicitly exported
# override (e.g. P3_ROUTE_FORCE arms) still takes effect.
FROZEN = {
    "DATASET": "olympics",
    "LLM_PROVIDER": "gemini",
    "GEMINI_MODEL": "gemini-2.5-flash",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "SEC_PRECHUNKED": "1",

    # ---- P2, frozen from the dev-30 sweep (docs/design/OLYMPICS_FROZEN_CONFIG.md) --------------
    # Winner "topk_20_c30": 0.867 strict / 4,997 tokens, vs the un-tuned k=10 baseline's
    # 0.633 / 6,135 — better accuracy AND fewer tokens. Selected purely on P2's own strict
    # pass; no arm was rejected for being good for P2.
    "CONTEXT_NUM_CHUNKS": "20",
    "CONTEXT_CHUNK_CHARS": "2000",
    "P2_SEARCH": "similarity",
    "P2_COMPRESS": "1",
    "P2_COMPRESS_SENTS": "30",
    "P2_ADAPTIVE_K": "0",        # adapt_wide scored 0.767 < 0.867 flat k=20; not adopted

    # ---- P3, pinned so the P2 budget change cannot leak into it ----------------------------
    # CONTEXT_NUM_CHUNKS/CONTEXT_CHUNK_CHARS are shared config, and P3 defaults MAX_CTX_CHUNKS/
    # MAX_CTX_CHARS off them. Raising P2 to k=20 would therefore have silently doubled P3's
    # chunk ceiling too. These pins hold P3 at exactly the dev3 configuration it was measured
    # at, so the k=20 decision is a change to the BASELINE ONLY and P3 gets no budget increase.
    # This deliberately breaks the contract's "same ceiling for both": the baseline is now
    # allowed a LARGER retrieval budget than P3. That asymmetry favours P2 and is declared.
    "MAX_CONTEXT_CHUNKS": "10",
    "MAX_CHUNK_CHARS": "2000",
    "P3_GROUP_ROW_CAP": "60",
    "P3_VEC_K": "10",
    "P3_LOOKUP_K": "6",
}


def main() -> int:
    argv = sys.argv[1:]
    # `--split dev|test` is spelled in Olympics terms; the harness wants the registry key.
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--split" and i + 1 < len(argv):
            s = argv[i + 1]
            out += ["--split", s if s.startswith("olympics_") else f"olympics_{s}"]
            i += 2
            continue
        out.append(argv[i])
        i += 1
    if "--out-dir" not in out:
        out += ["--out-dir", "results/olympics"]
    if "--p3-module" not in out:
        out += ["--p3-module", P3_MODULE]

    env = dict(os.environ)
    for k, v in FROZEN.items():
        env.setdefault(k, v)
    print("[olympics-bench] frozen env:")
    for k in FROZEN:
        print(f"    {k}={env[k]}" + ("" if env[k] == FROZEN[k] else "   (OVERRIDDEN)"))
    print(f"[olympics-bench] {HARNESS.name} {' '.join(out)}")
    return subprocess.call([sys.executable, str(HARNESS), *out], cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
