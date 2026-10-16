"""Pipeline-3 (GraphRAG) tuning on the Olympics DEV 30.

WHAT THERE IS TO TUNE — AND THE ANSWER IS "ALMOST NOTHING"
    P3 scores 29/30 on dev, and the one miss (pub-099) is the deliberately ambiguous
    (venue, date) key that the branch is designed NOT to tie-break. So there is no accuracy
    headroom to tune for, and tuning for it would mean tuning against the single question the
    design intentionally declines to answer. The remaining objective is therefore: can tokens
    come down with NO accuracy cost?

    Measured before running anything: the largest (sport, games) group in the whole corpus has
    43 events, and P3_GROUP_ROW_CAP is 60. The cap cannot bind. Lowering it to 30 truncates 6
    corpus groups and to 20 truncates 14 — and for the superlative branch truncation can drop
    the argmax row itself, i.e. it buys a few tokens by risking the answer. These arms exist to
    show that empirically rather than argue it.

    P3_VEC_K and P3_LOOKUP_K drive the vector fallback, which fires only when a branch fails to
    resolve. branch_resolved is True on 30/30 dev questions, so both are inert on dev and are
    NOT tuned here; they stay at their as-built defaults and `--check-inert` proves it.

    DEV ONLY.

Run:
    python "build/olympics/p3_tune_olympics.py" --check-inert
    python "build/olympics/p3_tune_olympics.py"
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT = ROOT / "results/olympics/p3_tune"
BENCH = ROOT / "build/olympics/run_olympics_benchmark.py"
EVAL = ROOT / "build/olympics/evaluate_olympics.py"

ARMS = {
    "base":    {},                              # P3_GROUP_ROW_CAP=60, VEC_K=10, LOOKUP_K=6
    "cap_30":  {"P3_GROUP_ROW_CAP": "30"},
    "cap_20":  {"P3_GROUP_ROW_CAP": "20"},
}


def check_inert() -> None:
    """Prove the claims this file rests on, from files, before any arm runs."""
    import collections
    ev = json.load(open(ROOT / "pipeline3_olympics/index/events.json"))
    it = ev.values() if isinstance(ev, dict) else ev
    g = collections.Counter()
    for e in it:
        g[(e.get("sport_key") or e.get("sport"), e.get("games_id") or e.get("games"))] += 1
    sizes = sorted(g.values())
    print(f"[inert] (sport, games) groups: {len(sizes)}   largest: {max(sizes)}")
    for cap in (60, 30, 20):
        print(f"[inert]   groups truncated at cap={cap}: {sum(1 for s in sizes if s > cap)}")
    print(f"[inert] => P3_GROUP_ROW_CAP=60 binds on {sum(1 for s in sizes if s > 60)} groups "
          f"(it is a dead knob at this corpus size)")

    d = json.load(open(ROOT / "results/olympics/bench_olympics_dev_dev3_run1.json"))
    rows = d["pipelines"]["graphrag"]
    res = collections.Counter(r.get("branch_resolved") for r in rows)
    print(f"[inert] branch_resolved on dev: {dict(res)}  "
          f"=> vector fallback fires on {res[False]} / {len(rows)} questions")
    print(f"[inert] => P3_VEC_K and P3_LOOKUP_K are inert on dev; left at defaults, not tuned")


def run_arm(name: str) -> dict:
    env = {**os.environ, **ARMS[name]}
    tag = f"p3t_{name}"
    bench = OUT / f"bench_olympics_dev_{tag}_run1.json"
    ev = OUT / f"eval_{tag}.json"
    if not ev.exists():
        t0 = time.perf_counter()
        subprocess.run([PY, str(BENCH), "--split", "dev", "--runs", "1", "--pipelines", "3",
                        "--tag", tag, "--out-dir", str(OUT)], env=env, check=True, cwd=str(ROOT))
        subprocess.run([PY, str(EVAL), str(bench), "--out", str(ev), "--skip-cite-judge"],
                       env=env, check=True, cwd=str(ROOT))
        print(f"  [{name}] {time.perf_counter()-t0:.0f}s", flush=True)
    a = json.load(open(ev))["pipelines"]["graphrag"]["aggregate"]
    b = json.load(open(bench))["aggregates"]["graphrag"]
    rows = json.load(open(bench))["pipelines"]["graphrag"]
    return {"arm": name, "strict": a["strict_pass_rate"], "graded": a["graded_quality_mean"],
            "bert": a["bertscore_f1_mean"], "tokens": a["avg_total_inference_tokens"],
            "recall": a["retrieval_recall_mean"], "latency": b.get("avg_latency_seconds"),
            "tg_live": b.get("tg_live_rate"),
            "resolved": sum(1 for r in rows if r.get("branch_resolved"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-inert", action="store_true")
    a = ap.parse_args()
    check_inert()
    if a.check_inert:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_arm(n) for n in ARMS]
    print("\n" + "=" * 92)
    print(f"{'arm':10} {'strict':>7} {'graded':>7} {'bert':>7} {'tokens':>8} {'recall':>7} "
          f"{'lat(s)':>7} {'tg_live':>8} {'resolved':>9}")
    print("-" * 92)
    for r in rows:
        print(f"{r['arm']:10} {r['strict']:>7.3f} {r['graded']:>7.3f} {r['bert']:>7.3f} "
              f"{r['tokens']:>8.0f} {r['recall']:>7.3f} {str(r['latency']):>7} "
              f"{r['tg_live']:>8.2f} {r['resolved']:>9}")
    json.dump(rows, open(OUT / "p3_tune_rows.json", "w"), indent=2)
    print(f"\n[p3-tune] wrote {OUT}/p3_tune_rows.json")


if __name__ == "__main__":
    main()
