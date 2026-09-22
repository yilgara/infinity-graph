"""Phase 1: build NetworkX graph objects from the raw CSVs and persist them.

Produces:
  data/processed/unimodal.graphml   - shipped 327-node weighted co-appearance graph
  data/processed/bimodal.graphml    - character<->comic bipartite graph
"""
import networkx as nx
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MARVEL_DIR = BASE / "data/marvel"
OUT_DIR = BASE / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_unimodal():
    nodes = pd.read_csv(MARVEL_DIR / "marvel-unimodal-nodes.csv")
    edges = pd.read_csv(MARVEL_DIR / "marvel-unimodal-edges.csv")

    G = nx.Graph()
    G.add_nodes_from(nodes["Id"].tolist())
    for _, row in edges.iterrows():
        G.add_edge(row["Source"], row["Target"], weight=int(row["Weight"]))

    assert G.number_of_nodes() == 327, G.number_of_nodes()
    assert G.number_of_edges() == 9891, G.number_of_edges()
    return G


def build_bimodal():
    bnodes = pd.read_csv(MARVEL_DIR / "bimodal/marvel-bimodal-nodes.csv")
    bedges = pd.read_csv(MARVEL_DIR / "bimodal/marvel-bimodal-edges.csv")

    B = nx.Graph()
    heroes = bnodes[bnodes["type"] == "hero"]["ID"].tolist()
    comics = bnodes[bnodes["type"] == "comic"]["ID"].tolist()
    B.add_nodes_from(heroes, bipartite="hero")
    B.add_nodes_from(comics, bipartite="comic")

    node_types = dict(zip(bnodes["ID"], bnodes["type"]))
    dropped = 0
    for _, row in bedges.iterrows():
        s, t = row["Source"], row["Target"]
        # bimodal edges should always be hero<->comic; guard against malformed rows
        if node_types.get(s) == node_types.get(t):
            dropped += 1
            continue
        B.add_edge(s, t)

    print(f"bimodal: dropped {dropped} malformed (same-type) edges out of {len(bedges)}")
    assert nx.is_bipartite(B)
    return B, heroes, comics


def report(G, name):
    print(f"\n=== {name} ===")
    print(f"nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    print(f"connected components: {len(comps)}; giant component size: {len(comps[0])} "
          f"({100*len(comps[0])/G.number_of_nodes():.1f}% of nodes)")
    degrees = [d for _, d in G.degree()]
    print(f"degree: min={min(degrees)}, max={max(degrees)}, mean={sum(degrees)/len(degrees):.2f}")


if __name__ == "__main__":
    G_uni = build_unimodal()
    report(G_uni, "Unimodal (shipped)")
    nx.write_graphml(G_uni, OUT_DIR / "unimodal.graphml")

    B, heroes, comics = build_bimodal()
    print(f"\n=== Bimodal ===")
    print(f"heroes: {len(heroes)}, comics: {len(comics)}, edges: {B.number_of_edges()}")
    hero_degrees = [B.degree(h) for h in heroes]
    print(f"hero degree (# comics appeared in): min={min(hero_degrees)}, max={max(hero_degrees)}, "
          f"mean={sum(hero_degrees)/len(hero_degrees):.2f}")
    comic_degrees = [B.degree(c) for c in comics]
    print(f"comic degree (# heroes in issue): min={min(comic_degrees)}, max={max(comic_degrees)}, "
          f"mean={sum(comic_degrees)/len(comic_degrees):.2f}")
    nx.write_graphml(B, OUT_DIR / "bimodal.graphml")

    print(f"\nSaved graphs to {OUT_DIR}")
