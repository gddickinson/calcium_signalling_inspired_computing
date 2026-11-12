"""Reference numbers for web/static/store_engine.js from multistore_computing (np.random.seed).

Run: python scripts/make_store_fixture.py [--write]
"""
import numpy as np

from fixture_common import main

from multistore_computing import HybridComputing, MultiStoreComputer

NAME = "store"


def summary(c):
    return {name: {"activation": s.activation, "contents_sum": s.contents.sum(axis=0)} for name, s in c.stores.items()}


def run(seed):
    np.random.seed(seed)
    c = MultiStoreComputer()
    writes = [c.write_data(d, priority=p) for d, p in [(0.9, "high"), ([0.5, 0.2, 0.7], "medium"),
                                                       (np.random.random(50), "low"), (0.95, "high")]]
    c.run_simulation(steps=25)
    transfers = [len(t) for t in c.cross_store_transfers]
    q = [{"source": r["source"], "confidence": r["confidence"], "result": r["result"]} for r in c.query(0.9)]
    state = summary(c)
    h = HybridComputing(c).hierarchical_memory([0.8, [0.2, 0.4, 0.6], 0.5, [0.1, 0.3, 0.5], 0.3], [0.8, 0.5, 0.3], steps=20)
    hm = {"writes": [[w["priority"], bool(w["success"]), w["store"]] for w in h["write_results"]],
          "retrievals": [[{"source": r["source"], "confidence": r["confidence"], "result": r["result"]} for r in rr["results"]]
                         for rr in h["retrieval_results"]]}
    return {"seed": seed, "writes": [[bool(ok), s] for ok, s in writes], "transfers": transfers, "query": q,
            "state": state, "hm": hm}


def fixture():
    return {"runs": [run(42), run(11)]}


if __name__ == "__main__":
    main(NAME, fixture)
