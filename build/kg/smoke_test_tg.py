"""Post-ingest smoke test of the TigerGraph 'finance' graph.

Verifies the graph is ready for P3 by exercising the exact queries P3 uses:
  - agg_audited_by('kpmg llp')      -> KPMG-audited companies (Tier C)
  - agg_uses_supplier('tsmc')       -> TSMC foundry users (Tier C)
  - agg_discusses_risk('climate change')
  - vector_search(embedding, k)     -> semantic chunk retrieval (the vector DB)
  - typed_neighbors(['apple inc. (aapl)']) -> curated one-hop expansion (bridge)
Uses GSQL/installed queries, not getVertexCount / getVerticesById (both unreliable here).
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
import pyTigerGraph as tg


def main():
    conn = tg.TigerGraphConnection(host=os.getenv("TIGERGRAPH_HOST"), graphname="finance",
                                   gsqlSecret=os.getenv("TIGERGRAPH_SECRET") or None,
                                   username=os.getenv("TIGERGRAPH_USERNAME"),
                                   password=os.getenv("TIGERGRAPH_PASSWORD"))
    conn.getToken(os.getenv("TIGERGRAPH_SECRET"))
    ok = True

    def show(label, fn):
        nonlocal ok
        try:
            r = fn()
            print(f"  ✓ {label}: {r}")
        except Exception as e:
            ok = False
            print(f"  ✗ {label}: ERROR {str(e)[:120]}")

    print("Aggregation queries (Tier C):")
    show("agg_audited_by(kpmg llp)",
         lambda: sorted(conn.runInstalledQuery("agg_audited_by", {"target_name": "kpmg llp"})[0]["companies"]))
    show("agg_uses_supplier(tsmc)",
         lambda: sorted(conn.runInstalledQuery("agg_uses_supplier", {"target_name": "tsmc"})[0]["companies"]))
    show("agg_discusses_risk(climate change) count",
         lambda: len(conn.runInstalledQuery("agg_discusses_risk", {"target_name": "climate change"})[0]["companies"]))

    print("Bridge / typed expansion:")
    show("typed_neighbors(apple inc. (aapl))",
         lambda: conn.runInstalledQuery("typed_neighbors", {"seed_names": ["apple inc. (aapl)"]})[0]["neighbors"][:5])

    print("Vector search (TigerGraph as vector DB):")
    def vec():
        from embeddings.embedder import get_embedder
        qv = get_embedder().embed_query("total net sales for fiscal year 2025")
        r = conn.runInstalledQuery("vector_search", {"query_vec": qv, "k": 5})
        ids = []
        for row in r:
            for v in (row.get("vs") or []):
                ids.append(v.get("v_id"))
        return ids[:5]
    show("vector_search(top-5 chunk ids)", vec)

    print("\n" + ("✅ TigerGraph 'finance' READY for Pipeline 3" if ok
                  else "⚠ some queries failed — check ingest completed"))


if __name__ == "__main__":
    main()
