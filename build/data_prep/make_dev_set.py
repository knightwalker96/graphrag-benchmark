"""Author the DEV question set (for tuning only; disjoint from the held-out 50).

Design goal (per PS 'Tuning and Test-Set Isolation'): a dev set that mirrors the
official eval set's tier structure and question shapes, but on DIFFERENT companies
/ facts so no held-out answer leaks into tuning. Same 4 tiers, similar proportions.

All answers were grounded against the actual filings in data/data/ at
authoring time (auditors, sectors, events verified by grep). We still spot-check
answers empirically before trusting dev metrics — dev is for RELATIVE tuning, not
for claiming absolute accuracy.

Output: datasets/sec/dev_set.jsonl  (split:"dev")
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path("datasets/sec/dev_set.jsonl")

DEV = [
    # --- Tier A: single-hop controls (single-doc lookup) ---
    ("DEV01", "A", "1", "single-doc",
     "What was Microsoft's total revenue for fiscal year 2025?", "MSFT 10-K",
     "$281,724 million (~$281.7B)"),
    ("DEV02", "A", "1", "single-doc",
     "On what date does Apple's fiscal year 2025 end?", "AAPL 10-K",
     "September 27, 2025"),
    ("DEV03", "A", "1", "single-doc",
     "Who is The Home Depot's independent registered public accounting firm?", "HD DEF14A",
     "KPMG LLP"),
    ("DEV04", "A", "1", "single-doc",
     "Which cloud/e-commerce segment is Amazon's largest source of operating income?", "AMZN 10-K",
     "Amazon Web Services (AWS)"),
    ("DEV05", "A", "1", "single-doc",
     "Who is UnitedHealth Group's independent auditor?", "UNH DEF14A",
     "Deloitte & Touche LLP"),

    # --- Tier B: two-hop bridges (event doc -> other doc) ---
    ("DEV06", "B", "2", "few-docs",
     "The company whose 10-K names Deloitte as auditor and whose fiscal year ends June 30, 2025 — what is that company?",
     "MSFT DEF14A + 10-K", "Microsoft"),
    ("DEV07", "B", "2", "few-docs",
     "Goldman Sachs' independent auditor is whom, and in what sector is Goldman Sachs classified?",
     "GS DEF14A + manifest", "PricewaterhouseCoopers LLP; Financials"),
    ("DEV08", "B", "2", "few-docs",
     "Which had the higher fiscal-year total revenue: Microsoft or Alphabet?",
     "MSFT & GOOGL 10-K", "Alphabet ($350,018M vs Microsoft $281,724M)"),
    ("DEV09", "B", "2", "few-docs",
     "Costco's auditor is whom, and in which sector is Costco classified?",
     "COST DEF14A + manifest", "KPMG LLP; Consumer Staples"),
    ("DEV10", "B", "2", "few-docs",
     "Which spent more on research and development in its latest fiscal year: Microsoft or Meta?",
     "MSFT & META 10-K", "Meta ($57,372M vs Microsoft R&D)"),

    # --- Tier C: multi-hop aggregation / intersection (corpus scan) ---
    ("DEV11", "C", "3+", "corpus-scan",
     "List the dataset companies in the Energy sector.", "manifest + 10-Ks",
     "CVX, XOM, COP (Energy-sector tickers in the dataset)"),
    ("DEV12", "C", "3+", "corpus-scan",
     "Which dataset companies are audited by Deloitte and classified in Health Care?",
     "DEF14A + sector", "BMY, UNH"),
    ("DEV13", "C", "3+", "corpus-scan",
     "Which Information Technology companies in the dataset are audited by Deloitte?",
     "DEF14A + sector", "MSFT (Microsoft)"),
    ("DEV14", "C", "3+", "corpus-scan",
     "Which dataset companies filed an 8-K reporting a CEO or executive-officer transition (Item 5.02)?",
     "8-Ks", "Companies with an Item 5.02 officer-change 8-K, e.g. AAPL, AMGN"),
    ("DEV15", "C", "3+", "corpus-scan",
     "How many dataset companies does Deloitte audit?", "DEF14A (all)",
     "24 companies"),
    ("DEV16", "C", "3+", "corpus-scan",
     "Which Financials-sector companies in the dataset are audited by PricewaterhouseCoopers?",
     "DEF14A + sector", "AIG, AXP, BAC, GS, JPM, MA, PYPL"),
    ("DEV17", "C", "3+", "corpus-scan",
     "Which dataset companies are audited by Ernst & Young and classified in Information Technology?",
     "DEF14A + sector", "AMD, AAPL, INTC, INTU, ORCL, PLTR (Ernst & Young-audited IT companies)"),

    # --- Tier D: cross-document & temporal ---
    ("DEV18", "D", "2", "few-docs",
     "Apple filed an 8-K on April 20, 2026. What corporate event does it report?",
     "AAPL 8-K 2026-04-20", "CEO transition: Tim Cook to Executive Chair, John Ternus appointed CEO"),
    ("DEV19", "D", "2", "few-docs",
     "The company that appointed John Ternus as CEO — who is its independent auditor?",
     "AAPL 8-K + DEF14A", "Apple; Ernst & Young LLP"),
    ("DEV20", "D", "2", "few-docs",
     "Amgen announced a CFO transition in a 2026 8-K — who is the incoming CFO?",
     "AMGN 8-K 2026-05-19", "Thomas Dittrich"),

    # --- A few extra single/two-hop for router-tuning coverage ---
    ("DEV21", "A", "1", "single-doc",
     "Who is the Chief Executive Officer named in Amazon's proxy statement?", "AMZN DEF14A",
     "Andrew R. Jassy"),
    ("DEV22", "B", "2", "few-docs",
     "Which Energy-sector company in the dataset reported the highest total revenue?",
     "Energy 10-Ks", "ExxonMobil (XOM), $332,238M total revenues and other income"),
    ("DEV23", "C", "3+", "corpus-scan",
     "Which dataset companies mention artificial intelligence (AI) as a growth area or risk in their 10-K?",
     "10-Ks", "Many companies flag AI as a growth area or risk (broad corpus scan)"),
    ("DEV24", "D", "2", "few-docs",
     "Home Depot's auditor is whom, and in which sector is Home Depot classified?",
     "HD DEF14A + manifest", "KPMG LLP; Consumer Discretionary"),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for qid, tier, hop, fanout, q, src, ans in DEV:
            f.write(json.dumps({
                "id": qid, "tier": tier, "hop": hop, "fanout": fanout,
                "question": q, "source_docs": src, "answer": ans, "split": "dev",
            }, ensure_ascii=False) + "\n")
    by_tier: dict[str, int] = {}
    for r in DEV:
        by_tier[r[1]] = by_tier.get(r[1], 0) + 1
    print(f"[dev-set] wrote {len(DEV)} dev questions -> {OUT}")
    print(f"[dev-set] by tier: {dict(sorted(by_tier.items()))}")


if __name__ == "__main__":
    main()
