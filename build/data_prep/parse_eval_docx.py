"""Parse the official eval questions .docx into a structured JSONL.

The docx is a table: ID | Hop | Fan-out/benefit | Question | Source doc(s) | Answer key,
preceded by 'Tier X — ...' header rows. We walk paragraphs in order, track the
current tier, and group every run of 6 cell-paragraphs starting at an 'EQnn' id
into one question record.

Output: datasets/sec/eval_set.jsonl  (held-out official set — never tuned on)
Each line: {id, tier, hop, fanout, question, source_docs, answer, split:"test"}
"""
from __future__ import annotations
import json
import re
import zipfile
from pathlib import Path

DOCX = Path("data/Eval Questions/eval_questions.docx")
OUT = Path("datasets/sec/eval_set.jsonl")


def _paragraphs(docx: Path) -> list[str]:
    z = zipfile.ZipFile(docx)
    xml = z.read("word/document.xml").decode("utf-8")
    paras = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
    out = []
    for p in paras:
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S)
        t = "".join(texts)
        # unescape the few xml entities that appear (&amp; etc.)
        t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        if t.strip():
            out.append(t.strip())
    return out


def main() -> None:
    paras = _paragraphs(DOCX)
    records = []
    tier = ""
    i = 0
    tier_re = re.compile(r"^Tier\s+([A-D])\b\s*[—-]\s*(.*)$")
    eq_re = re.compile(r"^EQ\d{2,}$")

    while i < len(paras):
        p = paras[i]
        m = tier_re.match(p)
        if m:
            tier = m.group(1)
            i += 1
            continue
        if eq_re.match(p):
            # A question row = 6 consecutive cell paragraphs:
            # [id, hop, fanout, question, source_docs, answer]
            cells = paras[i:i + 6]
            if len(cells) == 6:
                qid, hop, fanout, question, source, answer = cells
                records.append({
                    "id": qid,
                    "tier": tier,
                    "hop": hop,
                    "fanout": fanout,
                    "question": question,
                    "source_docs": source,
                    "answer": answer,
                    "split": "test",
                })
                i += 6
                continue
        i += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_tier: dict[str, int] = {}
    for r in records:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    print(f"[parse-eval] wrote {len(records)} questions -> {OUT}")
    print(f"[parse-eval] by tier: {dict(sorted(by_tier.items()))}")
    # sanity: show first + last
    if records:
        print(f"[parse-eval] first: {records[0]['id']} {records[0]['question'][:60]}")
        print(f"[parse-eval] last:  {records[-1]['id']} {records[-1]['question'][:60]}")


if __name__ == "__main__":
    main()
