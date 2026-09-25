"""Build all three character-character models from the raw source CSVs, plus
the raw bipartite bimodal graph they're derived from.

  #1 shipped unimodal              -- not built, just loaded as-is
                                       (data/marvel/marvel-unimodal-*.csv)
  #2 our Jaccard (curated)         -- >=5-shared-comics cutoff, 2,396 chars.
                                       Main network used throughout the
                                       analysis (centrality/community/
                                       robustness), alongside #1.
  #3 our Jaccard (full, uncurated) -- no cutoff, all 6,439 chars. Diagnostic
                                       only: checks how much #2's curation
                                       inflates clustering/z-scores versus
                                       the full population.

#2 and #3 share the same projection logic (build_projection()) and differ
only by the min_shared_comics threshold. Raw co-occurrence counts are
computed as an intermediate for both (they're the numerator of the Jaccard
ratio, and let us validate the projection against the shipped file) but
aren't saved as their own model -- validation showed the raw-count
projection over the full character set reproduces the shipped 327-character
network almost exactly (81% exact edge-weight match, mean abs error 0.338),
so it would be redundant as a fourth model.

Known data-cleaning steps applied (see report for justification):
  - 'SPIDER-MAN / PETER PARKER' appears in bimodal edges but is missing from
    bimodal-nodes.csv (a typo'd duplicate 'PARKERKER' with degree 0 exists
    instead). Patched in as type='hero' so Spider-Man isn't silently dropped.
  - Character names differ in case and truncation length between the
    unimodal and bimodal files (e.g. bimodal 'MARVEL GIRL / JEAN GRE' vs
    unimodal 'Marvel Girl / Jean Grey'). Matched via case-insensitive exact
    match, falling back to bidirectional prefix match on the shorter
    string's length -- needed to validate #2 against the shipped file.
  - A single garbage bimodal node literally named 'M' creates a spurious
    prefix match for every unimodal name; excluded explicitly.

Outputs (data/processed/):
  unimodal.graphml               - model #1
  bimodal.graphml                - raw bipartite cache (not a model itself)
  projection_jaccard.graphml     - model #2
  projection_full_uncurated.graphml - model #3
"""
import networkx as nx
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
MARVEL_DIR = BASE / "data/marvel"
OUT_DIR = BASE / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_MATCHABLE_ID_LEN = 2  # single-character bimodal IDs (e.g. 'M', 'A', 'T', 'P') are
                          # excluded from name matching: any single letter is trivially
                          # a prefix of nearly every longer name, causing false matches
SPIDER_MAN_FIX = "SPIDER-MAN / PETER PARKER"  # present in edges, missing from nodes.csv
MIN_SHARED_COMICS = 5  # matches shipped unimodal file's own curation cutoff


# --- Model #1: shipped unimodal (just loaded, not built) -------------------

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


# --- Raw bimodal graph (cache only, not a model) ----------------------------

def load_bimodal():
    bnodes = pd.read_csv(MARVEL_DIR / "bimodal/marvel-bimodal-nodes.csv")
    bedges = pd.read_csv(MARVEL_DIR / "bimodal/marvel-bimodal-edges.csv")
    node_types = dict(zip(bnodes["ID"], bnodes["type"]))
    node_types[SPIDER_MAN_FIX] = "hero"
    return bnodes, bedges, node_types


def build_bimodal_graph(bnodes, bedges, node_types):
    B = nx.Graph()
    heroes = bnodes[bnodes["type"] == "hero"]["ID"].tolist()
    comics = bnodes[bnodes["type"] == "comic"]["ID"].tolist()
    B.add_nodes_from(heroes, bipartite="hero")
    B.add_nodes_from(comics, bipartite="comic")

    dropped = 0
    for s, t in zip(bedges["Source"], bedges["Target"]):
        if node_types.get(s) == node_types.get(t):
            dropped += 1
            continue
        B.add_edge(s, t)
    print(f"bimodal: dropped {dropped} malformed (same-type) edges out of {len(bedges)}")
    assert nx.is_bipartite(B)
    return B, heroes, comics


def build_comic_groups(bedges, node_types):
    comic_to_heroes = defaultdict(set)
    hero_comics = defaultdict(set)
    dropped = 0
    for s, t in zip(bedges["Source"], bedges["Target"]):
        ts, tt = node_types.get(s), node_types.get(t)
        if ts == "hero" and tt == "comic":
            hero, comic = s, t
        elif ts == "comic" and tt == "hero":
            hero, comic = t, s
        else:
            dropped += 1
            continue
        comic_to_heroes[comic].add(hero)
        hero_comics[hero].add(comic)
    print(f"dropped {dropped} edges with unresolved/same node types")
    return comic_to_heroes, hero_comics


# --- Models #2 and #3: our Jaccard projection (curated / full) -------------

def build_projection(comic_to_heroes, hero_comics, min_shared_comics=MIN_SHARED_COMICS):
    raw_counts = defaultdict(int)
    for comic, heroes in comic_to_heroes.items():
        heroes = sorted(heroes)
        for h1, h2 in combinations(heroes, 2):
            raw_counts[(h1, h2)] += 1

    G = nx.Graph()
    G.add_nodes_from(hero_comics.keys())
    n_dropped_by_threshold = 0
    for (h1, h2), count in raw_counts.items():
        if count < min_shared_comics:
            n_dropped_by_threshold += 1
            continue
        union_size = len(hero_comics[h1] | hero_comics[h2])
        jaccard = count / union_size if union_size else 0.0
        G.add_edge(h1, h2, weight=count, jaccard=jaccard)
    print(f"dropped {n_dropped_by_threshold} pairs below min_shared_comics={min_shared_comics} "
          f"threshold (out of {len(raw_counts)} raw pairs)")
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def match_names(unimodal_ids, bimodal_hero_ids):
    """Map each unimodal character Id to its bimodal hero ID equivalent."""
    candidates_pool = [(h, h.lower()) for h in bimodal_hero_ids if len(h) >= MIN_MATCHABLE_ID_LEN]
    matched, unresolved = {}, []
    for uid in unimodal_ids:
        u_lower = uid.lower()
        exact = [h for h, hl in candidates_pool if hl == u_lower]
        if exact:
            matched[uid] = exact[0]
            continue
        cands = []
        for h, hl in candidates_pool:
            m = min(len(u_lower), len(hl))
            if u_lower[:m] == hl[:m]:
                cands.append(h)
        if len(cands) == 1:
            matched[uid] = cands[0]
        else:
            unresolved.append((uid, cands))
    return matched, unresolved


def validate_against_shipped(G_raw, name_map):
    unodes = pd.read_csv(MARVEL_DIR / "marvel-unimodal-nodes.csv")
    uedges = pd.read_csv(MARVEL_DIR / "marvel-unimodal-edges.csv")

    n_checked, n_exact, n_missing_edge, n_missing_node = 0, 0, 0, 0
    abs_errors = []
    for _, row in uedges.iterrows():
        u1, u2, w_shipped = row["Source"], row["Target"], row["Weight"]
        b1, b2 = name_map.get(u1), name_map.get(u2)
        if b1 is None or b2 is None:
            n_missing_node += 1
            continue
        n_checked += 1
        if G_raw.has_edge(b1, b2):
            w_ours = G_raw[b1][b2]["weight"]
            abs_errors.append(abs(w_ours - w_shipped))
            if w_ours == w_shipped:
                n_exact += 1
        else:
            n_missing_edge += 1

    print(f"\n=== Validation vs shipped unimodal ({len(uedges)} shipped edges) ===")
    print(f"both endpoints name-matched: {n_checked}")
    print(f"  exact weight match: {n_exact} ({100*n_exact/n_checked:.1f}%)")
    print(f"  edge missing entirely in our projection: {n_missing_edge}")
    print(f"  endpoint(s) unmatched by name: {n_missing_node}")
    if abs_errors:
        import numpy as np
        ae = np.array(abs_errors)
        print(f"  mean abs weight error: {ae.mean():.3f}, median: {np.median(ae):.1f}, max: {ae.max()}")
        print(f"  errors == 0: {(ae==0).sum()}, errors <=1: {(ae<=1).sum()}, errors > 5: {(ae>5).sum()}")


def describe(G, label):
    print(f"\n=== {label} ===")
    print(f"nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    print(f"connected components: {len(comps)}; giant component: {len(comps[0])} "
          f"({100*len(comps[0])/G.number_of_nodes():.1f}%)")
    if nx.get_edge_attributes(G, "jaccard"):
        jaccard_vals = [d["jaccard"] for _, _, d in G.edges(data=True)]
        print(f"jaccard weight: min={min(jaccard_vals):.4f}, max={max(jaccard_vals):.4f}, "
              f"mean={sum(jaccard_vals)/len(jaccard_vals):.4f}")


if __name__ == "__main__":
    # --- Model #1: shipped unimodal ---
    G1 = build_unimodal()
    describe(G1, "Model #1: shipped unimodal")
    nx.write_graphml(G1, OUT_DIR / "unimodal.graphml")

    # --- Raw bimodal graph (cache) ---
    bnodes, bedges, node_types = load_bimodal()
    B, heroes, comics = build_bimodal_graph(bnodes, bedges, node_types)
    print(f"\n=== Raw bimodal (cache, not a model) ===")
    print(f"heroes: {len(heroes)}, comics: {len(comics)}, edges: {B.number_of_edges()}")
    nx.write_graphml(B, OUT_DIR / "bimodal.graphml")

    # --- Models #2 and #3: our Jaccard projection (curated / full) ---
    comic_to_heroes, hero_comics = build_comic_groups(bedges, node_types)
    print(f"\ncomics with >=2 heroes: {sum(1 for h in comic_to_heroes.values() if len(h)>=2)} / {len(comic_to_heroes)}")

    G2 = build_projection(comic_to_heroes, hero_comics)
    describe(G2, "Model #2: our Jaccard projection (curated, >=5 shared comics)")

    unodes = pd.read_csv(MARVEL_DIR / "marvel-unimodal-nodes.csv")
    bimodal_heroes = [n for n, d in node_types.items() if d == "hero"]
    name_map, unresolved = match_names(unodes["Id"].tolist(), bimodal_heroes)
    print(f"\nname matching: {len(name_map)}/{len(unodes)} resolved, {len(unresolved)} unresolved")
    for u, c in unresolved:
        print(f"  UNRESOLVED: {u} -> {c}")
    validate_against_shipped(G2, name_map)

    nx.write_graphml(G2, OUT_DIR / "projection_jaccard.graphml")
    print(f"\nSaved model #2 to {OUT_DIR / 'projection_jaccard.graphml'}")

    G3 = build_projection(comic_to_heroes, hero_comics, min_shared_comics=1)
    describe(G3, "Model #3: our Jaccard projection (full, uncurated, diagnostic only)")
    nx.write_graphml(G3, OUT_DIR / "projection_full_uncurated.graphml")
    print(f"\nSaved model #3 to {OUT_DIR / 'projection_full_uncurated.graphml'}")
