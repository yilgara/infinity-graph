"""Single-node criticality via articulation points (cut vertices) -- a
different question from Phase 6's batch percolation curves: not "how does
the network degrade under a sequence of removals" but "is there any single
character whose removal alone fragments the network, and if so, how badly."

A node is an articulation point if removing it (alone) increases the number
of connected components. Uses NetworkX's linear-time (Hopcroft-Tarjan DFS)
implementation -- no need to test every node individually.

For each articulation point, "damage" = how many nodes end up outside the
new largest remaining component after removing it (i.e. everything that got
cut off, not just the 1 node removed itself).
"""
import networkx as nx
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROC_DIR = BASE / "data/processed"

NETWORK_FILES = {
    "shipped_unimodal": "unimodal.graphml",
    "our_jaccard": "projection_jaccard.graphml",
    "our_jaccard_full": "projection_full_uncurated.graphml",
}

# characters we specifically want to compare against, since they top every
# other centrality measure in this project (Phase 4)
WATCH_LIST = ["Captain America", "Spider-man / Peter Parker",
              "CAPTAIN AMERICA", "SPIDER-MAN / PETER PARKER"]


def giant_component(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(comps[0]).copy()


def damage_of(G, node):
    """Nodes left outside the new largest component after removing `node`."""
    H = G.copy()
    H.remove_node(node)
    if H.number_of_nodes() == 0:
        return 0, []
    pieces = sorted([len(c) for c in nx.connected_components(H)], reverse=True)
    return sum(pieces[1:]), pieces[:5]


def analyze(name, G):
    Gc = giant_component(G)
    n = Gc.number_of_nodes()
    aps = list(nx.articulation_points(Gc))
    print(f"\n=== {name} (n={n}) ===")
    print(f"articulation points: {len(aps)}")

    if not aps:
        print("No single character's removal fragments this network.")
        return pd.DataFrame(columns=["node", "damage", "degree", "top_pieces"])

    rows = []
    for ap in aps:
        damage, pieces = damage_of(Gc, ap)
        rows.append({"node": ap, "damage": damage, "degree": Gc.degree(ap),
                      "top_pieces": pieces})
    df = pd.DataFrame(rows).sort_values("damage", ascending=False).reset_index(drop=True)

    print(f"worst single removal: {df.iloc[0]['node']} "
          f"(damage={df.iloc[0]['damage']}, {100*df.iloc[0]['damage']/n:.2f}% of network)")
    print("\ntop 10 by damage:")
    for _, row in df.head(10).iterrows():
        print(f"  {row['node']:<30} damage={row['damage']:<5} degree={row['degree']:<5} "
              f"pieces={row['top_pieces']}")

    for watch in WATCH_LIST:
        if watch in df["node"].values:
            rank = df.index[df["node"] == watch][0] + 1
            row = df[df["node"] == watch].iloc[0]
            print(f"\n{watch}: rank #{rank}/{len(df)} by damage "
                  f"(damage={row['damage']}, degree={row['degree']})")
        elif watch in Gc.nodes():
            print(f"\n{watch}: NOT an articulation point (removal alone doesn't fragment anything)")

    return df


if __name__ == "__main__":
    for name, fname in NETWORK_FILES.items():
        G = nx.read_graphml(PROC_DIR / fname)
        df = analyze(name, G)
        if len(df):
            df.to_csv(PROC_DIR / f"articulation_points_{name}.csv", index=False)
            print(f"\nSaved to {PROC_DIR / f'articulation_points_{name}.csv'}")

    print("\n\nDone.")
