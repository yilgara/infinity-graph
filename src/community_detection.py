"""Phase 5: community detection, validated two ways --
  (1) modularity z-scored against the same degree-preserving null model
      ensemble used in Phase 3 (proves communities are real structure, not
      an artifact every graph has), and
  (2) a small manually-curated ground-truth list of real Marvel teams
      (Avengers / X-Men / Fantastic Four), checking whether Louvain actually
      groups them together.

Runs on all 3 models. Community detection uses unweighted Louvain (weight=
None), matching Phase 3/4's approach, for direct comparability across the
three networks and their null ensembles.
"""
import sys
import importlib.util
from pathlib import Path
from collections import defaultdict
import networkx as nx
import numpy as np
import random

BASE = Path(__file__).resolve().parent.parent
PROC_DIR = BASE / "data/processed"
SEED = 42

# --- reuse null-model rewiring + name-matching from earlier phases ---
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

s3 = _load_module("s3", BASE / "src/structural_stats.py")
bm = _load_module("bm", BASE / "src/build_models.py")

N_NULL_RUNS = 100

# Ground truth: manually curated known team rosters, unimodal-style names.
GROUND_TRUTH_TEAMS = {
    "Avengers": [
        "Captain America", "Iron Man / Tony Stark", "Thor / Dr. Donald Blak",
        "Vision", "Scarlet Witch / Wanda", "Hawk", "Wasp / Janet Van Dyne",
        "Ant-man / Dr. Henry J.",
    ],
    "X-Men": [
        "Wolverine / Logan", "Cyclops / Scott Summer", "Marvel Girl / Jean Grey",
        "Storm / Ororo Munroe S", "Beast / Henry &hank& P", "Colossus Ii / Peter Ra",
        "Angel / Warren Kenneth", "Nightcrawler / Kurt Wa", "Professor X / Charles",
        "Rogue  / ", "Gambit / Remy Lebeau", "Iceman / Robert Bobby",
    ],
    "Fantastic Four": [
        "Mr. Fantastic / Reed R", "Invisible Woman / Sue",
        "Thing / Benjamin J. Gr", "Human Torch / Johnny S",
    ],
}
ALL_GT_NAMES = [n for team in GROUND_TRUTH_TEAMS.values() for n in team]


def load_networks():
    G1 = nx.read_graphml(PROC_DIR / "unimodal.graphml")
    G2 = nx.read_graphml(PROC_DIR / "projection_jaccard.graphml")
    G3 = nx.read_graphml(PROC_DIR / "projection_full_uncurated.graphml")
    return {"shipped_unimodal": G1, "our_jaccard": G2, "our_jaccard_full": G3}


def giant_component(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(comps[0]).copy()


def detect_communities(G, seed=SEED):
    communities = nx.algorithms.community.louvain_communities(G, weight=None, seed=seed)
    modularity = nx.algorithms.community.modularity(G, communities, weight=None)
    return communities, modularity


def null_modularity_distribution(G, n_runs=N_NULL_RUNS):
    n_edges = G.number_of_edges()
    nswap = 10 * n_edges
    max_tries = nswap * 20
    rng = random.Random(SEED)

    mods = []
    for i in range(n_runs):
        R = G.copy()
        try:
            nx.double_edge_swap(R, nswap=nswap, max_tries=max_tries, seed=rng.randint(0, 10**9))
        except nx.NetworkXAlgorithmError:
            pass
        Rc = giant_component(R)
        _, mod = detect_communities(Rc, seed=rng.randint(0, 10**9))
        mods.append(mod)
        if (i + 1) % 20 == 0:
            print(f"    null run {i+1}/{n_runs}")
    return np.array(mods)


def node_to_community(communities):
    mapping = {}
    for i, comm in enumerate(communities):
        for node in comm:
            mapping[node] = i
    return mapping


def validate_ground_truth(node_community, name_map=None):
    print("\n--- Ground-truth team validation ---")
    for team, members in GROUND_TRUTH_TEAMS.items():
        rows = []
        for name in members:
            lookup = name_map.get(name) if name_map else name
            comm = node_community.get(lookup) if lookup else None
            rows.append((name, comm))
        found = [(n, c) for n, c in rows if c is not None]
        missing = [n for n, c in rows if c is None]
        if not found:
            print(f"{team}: no members found in this network")
            continue
        comm_counts = defaultdict(int)
        for _, c in found:
            comm_counts[c] += 1
        dominant_comm, dominant_count = max(comm_counts.items(), key=lambda x: x[1])
        print(f"\n{team}: {len(found)}/{len(members)} members found "
              f"(missing: {missing if missing else 'none'})")
        print(f"  split across {len(comm_counts)} distinct communities; "
              f"{dominant_count}/{len(found)} in the largest one (community #{dominant_comm})")
        for name, c in found:
            marker = " <-- majority" if c == dominant_comm else ""
            print(f"    {name:<35} community #{c}{marker}")


if __name__ == "__main__":
    networks = load_networks()

    bnodes, bedges, node_types = bm.load_bimodal()
    bimodal_heroes = [n for n, d in node_types.items() if d == "hero"]
    gt_name_map, unresolved = bm.match_names(ALL_GT_NAMES, bimodal_heroes)
    if unresolved:
        print(f"Warning: could not map to bimodal names: {unresolved}")

    for name, G in networks.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        Gc = giant_component(G)
        print(f"Giant component: {Gc.number_of_nodes()} nodes, {Gc.number_of_edges()} edges")

        communities, modularity = detect_communities(Gc)
        sizes = sorted([len(c) for c in communities], reverse=True)
        print(f"Louvain found {len(communities)} communities; sizes (top 10): {sizes[:10]}")
        print(f"Modularity: {modularity:.4f}")

        # model #3's rewiring is ~7x denser/slower (Phase 3 precedent: 100 runs
        # for #1/#2, 30 for #3 to keep runtime reasonable)
        n_runs = 30 if name == "our_jaccard_full" else N_NULL_RUNS
        print(f"Running {n_runs}-realization null ensemble for modularity z-score...")
        null_mods = null_modularity_distribution(Gc, n_runs=n_runs)
        mean, std = null_mods.mean(), null_mods.std()
        z = (modularity - mean) / std if std > 0 else float("nan")
        print(f"Null modularity: mean={mean:.4f}, std={std:.4f}, z={z:.2f}")

        node_community = node_to_community(communities)
        name_map = gt_name_map if name in ("our_jaccard", "our_jaccard_full") else None
        validate_ground_truth(node_community, name_map=name_map)

    print("\n\nDone.")
