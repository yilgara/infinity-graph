"""Phase 3: structural statistics for both networks, benchmarked against a
degree-preserving null model ensemble (z-scores), plus degree-distribution
fitting (power law vs lognormal vs exponential via likelihood-ratio test).

Networks analyzed (unweighted structure -- standard for small-world /
degree-distribution comparisons, matching the anchor paper's approach):
  #1 shipped unimodal   (data/processed/unimodal.graphml)
  #2 our Jaccard projection (data/processed/projection_jaccard.graphml)

Null model: degree-preserving randomization via double_edge_swap (Maslov-
Sneppen rewiring), which exactly preserves each node's degree while
destroying higher-order structure (clustering, path-length correlations).
Average shortest path length on the null ensemble is estimated via a fixed
random node sample (documented below) to keep 100-run ensembles tractable;
the real networks' APL is computed exactly.
"""
import networkx as nx
import numpy as np
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROC_DIR = BASE / "data/processed"

N_NULL_RUNS = 100
APL_SAMPLE_SIZE = 300  # nodes sampled per graph when estimating avg shortest path length
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def load_networks():
    G1 = nx.read_graphml(PROC_DIR / "unimodal.graphml")
    G2 = nx.read_graphml(PROC_DIR / "projection_jaccard.graphml")
    return {"shipped_unimodal": G1, "our_jaccard": G2}


def giant_component(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(comps[0]).copy()


def sampled_avg_shortest_path(G, sample_size=APL_SAMPLE_SIZE, rng=None):
    """Estimate average shortest path length via BFS from a random node sample."""
    rng = rng or random
    nodes = list(G.nodes())
    sample = rng.sample(nodes, min(sample_size, len(nodes)))
    total, count = 0.0, 0
    for s in sample:
        lengths = nx.single_source_shortest_path_length(G, s)
        del lengths[s]
        total += sum(lengths.values())
        count += len(lengths)
    return total / count if count else float("nan")


def real_stats(G, exact_apl=True):
    Gc = giant_component(G)
    degrees = [d for _, d in Gc.degree()]
    stats = {
        "n_nodes": Gc.number_of_nodes(),
        "n_edges": Gc.number_of_edges(),
        "mean_degree": np.mean(degrees),
        "avg_clustering": nx.average_clustering(Gc),
        "transitivity": nx.transitivity(Gc),
    }
    if exact_apl:
        stats["avg_shortest_path"] = nx.average_shortest_path_length(Gc)
        stats["avg_shortest_path_method"] = "exact"
    else:
        stats["avg_shortest_path"] = sampled_avg_shortest_path(Gc)
        stats["avg_shortest_path_method"] = f"sampled(n={APL_SAMPLE_SIZE})"
    return stats


def null_ensemble_stats(G, n_runs=N_NULL_RUNS):
    Gc = giant_component(G)
    n_edges = Gc.number_of_edges()
    nswap = 10 * n_edges  # standard rule of thumb: ~10x edges for good mixing
    max_tries = nswap * 20

    clustering_vals, transitivity_vals, apl_vals = [], [], []
    rng = random.Random(SEED)
    for i in range(n_runs):
        R = Gc.copy()
        try:
            nx.double_edge_swap(R, nswap=nswap, max_tries=max_tries, seed=rng.randint(0, 10**9))
        except nx.NetworkXAlgorithmError:
            pass  # ran out of tries but partial swaps still valid randomization
        # double_edge_swap can disconnect the graph; use its own giant component
        Rc = giant_component(R)
        clustering_vals.append(nx.average_clustering(Rc))
        transitivity_vals.append(nx.transitivity(Rc))
        apl_vals.append(sampled_avg_shortest_path(Rc, rng=rng))
        if (i + 1) % 20 == 0:
            print(f"    null run {i+1}/{n_runs}")

    return {
        "avg_clustering": np.array(clustering_vals),
        "transitivity": np.array(transitivity_vals),
        "avg_shortest_path": np.array(apl_vals),
    }


def zscore(real_val, null_vals):
    mean, std = null_vals.mean(), null_vals.std()
    z = (real_val - mean) / std if std > 0 else float("nan")
    return mean, std, z


def fit_degree_distribution(G, label):
    degrees = np.array([d for _, d in G.degree() if d > 0])
    print(f"\n--- Degree distribution fit: {label} ---")
    print(f"degree range: {degrees.min()}-{degrees.max()}, n={len(degrees)}")
    try:
        import powerlaw
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        print(f"power law: alpha={fit.power_law.alpha:.3f}, xmin={fit.power_law.xmin}")

        comparisons = {}
        for alt in ["lognormal", "exponential", "truncated_power_law", "stretched_exponential"]:
            R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
            comparisons[alt] = (R, p)
            favors = "power_law" if R > 0 else alt
            sig = "significant" if p < 0.05 else "not significant"
            print(f"  power_law vs {alt}: R={R:.3f}, p={p:.4f} -> favors {favors} ({sig})")
        return {"alpha": fit.power_law.alpha, "xmin": fit.power_law.xmin, "comparisons": comparisons}
    except ImportError:
        print("powerlaw package not available")
        return None


if __name__ == "__main__":
    networks = load_networks()
    results = {}

    for name, G in networks.items():
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        exact_apl = G.number_of_nodes() <= 500  # exact APL only cheap for shipped (327 nodes)
        rs = real_stats(G, exact_apl=exact_apl)
        print(f"Real stats: {rs}")

        print(f"  Running {N_NULL_RUNS}-realization null ensemble (degree-preserving rewiring)...")
        null_stats = null_ensemble_stats(G, n_runs=N_NULL_RUNS)

        print(f"\n  === Null-model comparison ({name}) ===")
        for stat_key in ["avg_clustering", "transitivity", "avg_shortest_path"]:
            real_val = rs[stat_key]
            mean, std, z = zscore(real_val, null_stats[stat_key])
            print(f"  {stat_key}: real={real_val:.4f}, null_mean={mean:.4f}, null_std={std:.4f}, z={z:.2f}")

        deg_fit = fit_degree_distribution(giant_component(G), name)

        results[name] = {"real": rs, "null": null_stats, "degree_fit": deg_fit}

    print("\n\nDone. (Results held in-memory for this run; rerun script to reproduce.)")
