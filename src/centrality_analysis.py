"""Phase 4: centrality analysis -- "true main character" by network
structure, vs. marketing/movie prominence.

Computes degree, betweenness, closeness, and eigenvector centrality on
models #1 (shipped unimodal) and #2 (our Jaccard, curated). All measures
are unweighted (binary connection structure), matching Phase 3's approach
and keeping the two networks' centrality rankings directly comparable to
each other -- weighting by raw count vs. Jaccard would otherwise conflate
"is centrality robust to network construction" with "is centrality robust
to edge-weight choice," two different questions.

Identifies "connectors" (high betweenness relative to degree -- bridge
between otherwise-separate groups) vs. "hubs" (high degree, proportionally
lower betweenness -- well-connected within their own cluster), and compares
structural rankings against a small, manually-justified list of
marketing/movie-prominent characters (MCU headline heroes as of the source
data's ~1999/2000 cutoff era plus perennial marketing icons).
"""
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

import build_models as bm

BASE = Path(__file__).resolve().parent.parent
PROC_DIR = BASE / "data/processed"
OUT_DIR = BASE / "data/processed"

TOP_N = 15

# Manually curated "marketing famous" list: characters with major solo
# franchises / widest mainstream name recognition, independent of their
# comic-book co-appearance structure. Justification: these are the
# characters most likely to appear on a general-audience "name a Marvel
# character" survey, primarily via films/merchandising rather than comic
# publication volume.
MARKETING_FAMOUS = [
    "Spider-man / Peter Parker",
    "Wolverine / Logan",
    "Hulk / Dr. Robert Bruce Banner",
    "Captain America",
    "Iron Man / Tony Stark",
    "Thor / Dr. Donald Blak",
    "Daredevil / Matt Murdo",
    "Punisher Ii / Frank Ca",
    "Storm / Ororo Munroe S",
    "Silver Surfer / Norrin",
]


def load_networks():
    G1 = nx.read_graphml(PROC_DIR / "unimodal.graphml")
    G2 = nx.read_graphml(PROC_DIR / "projection_jaccard.graphml")
    G3 = nx.read_graphml(PROC_DIR / "projection_full_uncurated.graphml")
    return {"shipped_unimodal": G1, "our_jaccard": G2, "our_jaccard_full": G3}


def giant_component(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(comps[0]).copy()


def compute_centralities(G):
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight=None)
    closeness = nx.closeness_centrality(G, distance=None)
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight=None)
    except Exception as e:
        print(f"  eigenvector_centrality_numpy failed ({e}), falling back to PageRank")
        eigenvector = nx.pagerank(G, weight=None)

    return pd.DataFrame({
        "degree": degree,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigenvector,
    })


def add_ranks(df):
    for col in ["degree", "betweenness", "closeness", "eigenvector"]:
        df[f"{col}_rank"] = df[col].rank(ascending=False, method="min").astype(int)
    return df


def print_top_n(df, measure, n=TOP_N):
    print(f"\n--- Top {n} by {measure} ---")
    top = df.sort_values(measure, ascending=False).head(n)
    for name, row in top.iterrows():
        print(f"  {row[f'{measure}_rank']:>3}. {name:<35} {measure}={row[measure]:.4f}  "
              f"(degree_rank={row['degree_rank']}, betweenness_rank={row['betweenness_rank']})")


def find_connectors(df, n=10):
    """Characters whose betweenness rank is much better (lower number) than
    their degree rank -- bridges between groups, not just locally popular."""
    df = df.copy()
    df["rank_gap"] = df["degree_rank"] - df["betweenness_rank"]
    print(f"\n--- Top {n} 'connectors' (betweenness rank >> degree rank) ---")
    top = df.sort_values("rank_gap", ascending=False).head(n)
    for name, row in top.iterrows():
        print(f"  {name:<35} degree_rank={int(row['degree_rank']):>4}  "
              f"betweenness_rank={int(row['betweenness_rank']):>4}  gap={int(row['rank_gap'])}")

    print(f"\n--- Top {n} 'pure hubs' (degree rank >> betweenness rank) ---")
    bottom = df.sort_values("rank_gap", ascending=True).head(n)
    for name, row in bottom.iterrows():
        print(f"  {name:<35} degree_rank={int(row['degree_rank']):>4}  "
              f"betweenness_rank={int(row['betweenness_rank']):>4}  gap={int(row['rank_gap'])}")


def rank_correlations(df):
    print("\n--- Spearman rank correlations between centrality measures ---")
    measures = ["degree", "betweenness", "closeness", "eigenvector"]
    for i, m1 in enumerate(measures):
        for m2 in measures[i+1:]:
            rho, p = spearmanr(df[m1], df[m2])
            print(f"  {m1} vs {m2}: rho={rho:.3f} (p={p:.2e})")


def marketing_comparison(df, name_map=None, name_variants=MARKETING_FAMOUS):
    """name_map: optional dict translating MARKETING_FAMOUS's unimodal-style
    names to this network's own node IDs (needed for model #2, which uses
    bimodal-style ALL-CAPS names)."""
    print(f"\n--- Marketing-famous characters: structural ranking ---")
    n_total = len(df)
    found = []
    for name in name_variants:
        lookup = name_map.get(name) if name_map else name
        if lookup is not None and lookup in df.index:
            row = df.loc[lookup]
            found.append(name)
            print(f"  {name:<35} degree_rank={int(row['degree_rank']):>4}/{n_total}  "
                  f"betweenness_rank={int(row['betweenness_rank']):>4}/{n_total}  "
                  f"eigenvector_rank={int(row['eigenvector_rank']):>4}/{n_total}")
        else:
            print(f"  {name:<35} NOT FOUND in this network")
    missing = set(name_variants) - set(found)
    if missing:
        print(f"  (missing: {missing} -- name likely doesn't match this network's naming convention)")


if __name__ == "__main__":
    networks = load_networks()
    all_results = {}

    # models #2 and #3 use bimodal-style ALL-CAPS names; build a translation
    # from MARKETING_FAMOUS's unimodal-style names for them
    bnodes, bedges, node_types = bm.load_bimodal()
    bimodal_heroes = [n for n, d in node_types.items() if d == "hero"]
    marketing_name_map, unresolved = bm.match_names(MARKETING_FAMOUS, bimodal_heroes)
    if unresolved:
        print(f"Warning: could not map to bimodal names: {unresolved}")

    for name, G in networks.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        Gc = giant_component(G)
        print(f"Computing centralities on giant component: {Gc.number_of_nodes()} nodes, {Gc.number_of_edges()} edges")

        df = compute_centralities(Gc)
        df = add_ranks(df)

        for measure in ["degree", "betweenness", "closeness", "eigenvector"]:
            print_top_n(df, measure)

        find_connectors(df)
        rank_correlations(df)

        name_map = marketing_name_map if name in ("our_jaccard", "our_jaccard_full") else None
        marketing_comparison(df, name_map=name_map)

        df.to_csv(OUT_DIR / f"centrality_{name}.csv")
        print(f"\nSaved full centrality table to {OUT_DIR / f'centrality_{name}.csv'}")

        all_results[name] = df

    print("\n\nDone.")
