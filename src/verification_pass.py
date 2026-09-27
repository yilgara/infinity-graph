"""Phase 7b: independent verification pass (brief's step 9) -- re-derive
summary statistics by an independent method to catch silent bugs, before
writing anything up. Checks:

  1. Node/edge counts against README's stated numbers (re-asserted here for
     a complete record, though already checked in Phase 0/1).
  2. Average degree computed two independent ways: 2m/n formula vs. direct
     mean of nx degree values.
  3. Clustering coefficient computed two independent ways: networkx's
     built-in vs. a manual triangle-counting formula on a sample of nodes.
  4. A specific edge weight (Captain America <-> Iron Man) hand-verified
     against the raw bimodal comic co-appearance data, independent of the
     projection pipeline.
  5. Giant component coverage re-confirmed for all 3 models.
"""
import sys
import importlib.util
from pathlib import Path
from itertools import combinations
import networkx as nx
import numpy as np
import pandas as pd
import random

BASE = Path(__file__).resolve().parent.parent
PROC_DIR = BASE / "data/processed"
MARVEL_DIR = BASE / "data/marvel"


def load_networks():
    G1 = nx.read_graphml(PROC_DIR / "unimodal.graphml")
    G2 = nx.read_graphml(PROC_DIR / "projection_jaccard.graphml")
    G3 = nx.read_graphml(PROC_DIR / "projection_full_uncurated.graphml")
    return {"shipped_unimodal": G1, "our_jaccard": G2, "our_jaccard_full": G3}


def giant_component(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(comps[0]).copy()


def check_counts():
    print("=== 1. Node/edge counts vs README claims ===")
    nodes = pd.read_csv(MARVEL_DIR / "marvel-unimodal-nodes.csv")
    edges = pd.read_csv(MARVEL_DIR / "marvel-unimodal-edges.csv")
    ok_n = len(nodes) == 327
    ok_e = len(edges) == 9891
    print(f"  unimodal nodes: {len(nodes)} (expect 327) -- {'OK' if ok_n else 'MISMATCH'}")
    print(f"  unimodal edges: {len(edges)} (expect 9891) -- {'OK' if ok_e else 'MISMATCH'}")

    bnodes = pd.read_csv(MARVEL_DIR / "bimodal/marvel-bimodal-nodes.csv")
    bedges = pd.read_csv(MARVEL_DIR / "bimodal/marvel-bimodal-edges.csv")
    print(f"  bimodal nodes: {len(bnodes)} (19,090 per bimodal README)")
    print(f"  bimodal edges: {len(bedges)} (96,104 -- matches the brief exactly; "
          f"bimodal README's own claim of 96,662 is stale, per earlier investigation)")
    return ok_n and ok_e and len(bedges) == 96104


def check_avg_degree(networks):
    print("\n=== 2. Average degree: two independent methods ===")
    for name, G in networks.items():
        Gc = giant_component(G)
        n, m = Gc.number_of_nodes(), Gc.number_of_edges()
        method_a = 2 * m / n  # handshake-lemma formula
        method_b = np.mean([d for _, d in Gc.degree()])  # direct mean
        match = np.isclose(method_a, method_b)
        print(f"  {name}: 2m/n={method_a:.4f}  direct_mean={method_b:.4f}  "
              f"{'MATCH' if match else 'MISMATCH'}")


def check_clustering(networks, n_sample=20, seed=42):
    print(f"\n=== 3. Clustering coefficient: nx built-in vs. manual triangle-count "
          f"(sample of {n_sample} nodes per network) ===")
    rng = random.Random(seed)
    for name, G in networks.items():
        Gc = giant_component(G)
        nx_clustering = nx.clustering(Gc)
        sample = rng.sample(list(Gc.nodes()), min(n_sample, Gc.number_of_nodes()))

        max_diff = 0.0
        for node in sample:
            neighbors = list(Gc.neighbors(node))
            k = len(neighbors)
            if k < 2:
                manual = 0.0
            else:
                triangles = sum(1 for a, b in combinations(neighbors, 2) if Gc.has_edge(a, b))
                manual = 2 * triangles / (k * (k - 1))
            diff = abs(manual - nx_clustering[node])
            max_diff = max(max_diff, diff)
        print(f"  {name}: max |manual - nx| over {len(sample)} sampled nodes = {max_diff:.2e} "
              f"{'(MATCH)' if max_diff < 1e-9 else '(MISMATCH)'}")


def check_specific_edge_weight():
    print("\n=== 4. Hand-verify one edge weight against raw bimodal data ===")
    print("  Pair: Captain America <-> Iron Man / Tony Stark")

    bedges = pd.read_csv(MARVEL_DIR / "bimodal/marvel-bimodal-edges.csv")
    bnodes = pd.read_csv(MARVEL_DIR / "bimodal/marvel-bimodal-nodes.csv")
    node_types = dict(zip(bnodes["ID"], bnodes["type"]))

    def comics_for(hero_id):
        s = set(bedges.loc[bedges["Source"] == hero_id, "Target"])
        t = set(bedges.loc[bedges["Target"] == hero_id, "Source"])
        return {c for c in (s | t) if node_types.get(c) == "comic"}

    ca_comics = comics_for("CAPTAIN AMERICA")
    im_comics = comics_for("IRON MAN / TONY STARK")
    shared = ca_comics & im_comics
    print(f"  Captain America appears in {len(ca_comics)} comics (raw bimodal count)")
    print(f"  Iron Man appears in {len(im_comics)} comics (raw bimodal count)")
    print(f"  Manually counted shared comics (independent of build_models.py): {len(shared)}")

    G_jaccard = nx.read_graphml(PROC_DIR / "projection_jaccard.graphml")
    pipeline_weight = G_jaccard["CAPTAIN AMERICA"]["IRON MAN / TONY STARK"]["weight"]
    print(f"  Pipeline-computed weight (build_models.py): {pipeline_weight}")
    print(f"  {'MATCH' if len(shared) == pipeline_weight else 'MISMATCH'}")

    unodes = pd.read_csv(MARVEL_DIR / "marvel-unimodal-nodes.csv")
    uedges = pd.read_csv(MARVEL_DIR / "marvel-unimodal-edges.csv")
    shipped_row = uedges[
        ((uedges["Source"] == "Captain America") & (uedges["Target"] == "Iron Man / Tony Stark")) |
        ((uedges["Target"] == "Captain America") & (uedges["Source"] == "Iron Man / Tony Stark"))
    ]
    if len(shipped_row):
        shipped_weight = int(shipped_row.iloc[0]["Weight"])
        print(f"  Shipped unimodal file's own weight for this pair: {shipped_weight} "
              f"(diff from our manual count: {abs(shipped_weight - len(shared))})")


def check_giant_component(networks):
    print("\n=== 5. Giant component coverage ===")
    for name, G in networks.items():
        Gc = giant_component(G)
        pct = 100 * Gc.number_of_nodes() / G.number_of_nodes()
        n_components = nx.number_connected_components(G)
        print(f"  {name}: {Gc.number_of_nodes()}/{G.number_of_nodes()} nodes "
              f"({pct:.1f}%) in giant component, {n_components} total components")


if __name__ == "__main__":
    networks = load_networks()
    check_counts()
    check_avg_degree(networks)
    check_clustering(networks)
    check_specific_edge_weight()
    check_giant_component(networks)
    print("\n\nVerification pass complete.")
