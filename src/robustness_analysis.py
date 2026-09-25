"""Phase 6: robustness / percolation analysis -- "whose removal would
fracture the story-network the most." Direct callback to Albert, Jeong &
Barabasi (2000), "Error and attack tolerance of complex networks", Nature.

For each network, simulates node removal under three strategies and tracks
giant-component size (as a fraction of the network's total nodes) after
each removal:
  - random order (averaged over N_RANDOM_TRIALS realizations)
  - descending degree (static ranking on the intact network, not adaptively
    recomputed after each removal -- matches the classic AJB approach)
  - descending betweenness (reusing Phase 4's precomputed centrality tables
    -- betweenness on the 6,403-node full network is expensive, no reason
    to compute it twice)

Efficiency note: naively removing nodes one at a time and recomputing
connected components from scratch is O(n * (V+E)), too slow at n=6,403.
Instead we simulate the REVERSE process -- adding nodes back in the
opposite order, using union-find to incrementally merge components -- which
gives the exact same giant-component-size curve in near-linear time
O((V+E) * alpha(V)). The largest component size is monotonically
non-decreasing as nodes are added, so a running max gives the full curve
after one pass.
"""
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
import random

BASE = Path(__file__).resolve().parent.parent
PROC_DIR = BASE / "data/processed"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_RANDOM_TRIALS = 20


def load_networks():
    G1 = nx.read_graphml(PROC_DIR / "unimodal.graphml")
    G2 = nx.read_graphml(PROC_DIR / "projection_jaccard.graphml")
    G3 = nx.read_graphml(PROC_DIR / "projection_full_uncurated.graphml")
    return {"shipped_unimodal": G1, "our_jaccard": G2, "our_jaccard_full": G3}


def giant_component(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(comps[0]).copy()


class UnionFind:
    __slots__ = ("parent", "size")

    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.size = {n: 1 for n in nodes}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]


def giant_component_curve(G, removal_order):
    """removal_order: list of all n nodes, in the order they get removed.
    Returns array of length n+1: giant_frac[k] = giant component size
    (as a fraction of n) after the first k nodes have been removed."""
    n = G.number_of_nodes()
    addition_order = list(reversed(removal_order))
    adj = {node: list(G.neighbors(node)) for node in G.nodes()}
    active = {node: False for node in G.nodes()}

    uf = UnionFind(G.nodes())
    max_size_after_adding = np.zeros(n + 1, dtype=np.int64)
    current_max = 0
    for i, node in enumerate(addition_order):
        active[node] = True
        for nb in adj[node]:
            if active[nb]:
                uf.union(node, nb)
        current_max = max(current_max, uf.size[uf.find(node)])
        max_size_after_adding[i + 1] = current_max

    giant_after_removal = np.array([max_size_after_adding[n - k] for k in range(n + 1)])
    return giant_after_removal / n


def critical_fraction(curve, threshold=0.5):
    """Fraction of nodes removed at which the giant component first drops
    below `threshold` of the original network size."""
    below = np.where(curve < threshold)[0]
    if len(below) == 0:
        return None
    return below[0] / (len(curve) - 1)


def run_percolation(name, G):
    Gc = giant_component(G)
    n = Gc.number_of_nodes()
    print(f"\n{'='*70}\n{name}  (n={n}, m={Gc.number_of_edges()})\n{'='*70}")

    nodes = list(Gc.nodes())

    # --- random removal, averaged over trials ---
    rng = random.Random(SEED)
    random_curves = []
    for t in range(N_RANDOM_TRIALS):
        order = nodes.copy()
        rng.shuffle(order)
        random_curves.append(giant_component_curve(Gc, order))
    random_curve = np.mean(random_curves, axis=0)

    # --- degree-targeted removal (static ranking on intact network) ---
    degree_order = [node for node, _ in sorted(Gc.degree(), key=lambda x: -x[1])]
    degree_curve = giant_component_curve(Gc, degree_order)

    # --- betweenness-targeted removal (reuse Phase 4's precomputed values) ---
    cent_path = PROC_DIR / f"centrality_{name}.csv"
    cent_df = pd.read_csv(cent_path, index_col=0)
    betweenness_order = cent_df.sort_values("betweenness", ascending=False).index.tolist()
    betweenness_order = [n_ for n_ in betweenness_order if n_ in set(nodes)]
    missing = set(nodes) - set(betweenness_order)
    if missing:
        betweenness_order += list(missing)  # shouldn't happen, but stay safe
    betweenness_curve = giant_component_curve(Gc, betweenness_order)

    results = {"random": random_curve, "degree": degree_curve, "betweenness": betweenness_curve}

    print("Critical fraction (giant component drops below 50% of original size):")
    for strat, curve in results.items():
        cf = critical_fraction(curve, 0.5)
        print(f"  {strat:<12}: {cf:.3f}" if cf is not None else f"  {strat:<12}: never drops below 50%")

    print("Critical fraction (giant component drops below 10% of original size):")
    for strat, curve in results.items():
        cf = critical_fraction(curve, 0.1)
        print(f"  {strat:<12}: {cf:.3f}" if cf is not None else f"  {strat:<12}: never drops below 10%")

    return results


if __name__ == "__main__":
    networks = load_networks()
    all_results = {}
    for name, G in networks.items():
        all_results[name] = run_percolation(name, G)
        for strat, curve in all_results[name].items():
            np.save(PROC_DIR / f"percolation_{name}_{strat}.npy", curve)

    print("\n\nDone. Curves saved to data/processed/percolation_*.npy")
