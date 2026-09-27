"""Phase 8: generate the report's figures.
  fig1_degree_distribution.pdf  - empirical CCDF + fitted distributions, all 3 networks
  fig2_robustness_curves.pdf    - giant component fraction vs fraction removed, all 3 networks, 3 strategies
  fig3_network_snapshot.pdf     - static layout of model #1, colored by community, sized by degree centrality
"""
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import powerlaw

BASE = Path(__file__).resolve().parent.parent
PROC_DIR = BASE / "data/processed"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NETWORK_FILES = {
    "shipped_unimodal": "unimodal.graphml",
    "our_jaccard": "projection_jaccard.graphml",
    "our_jaccard_full": "projection_full_uncurated.graphml",
}
NETWORK_LABELS = {
    "shipped_unimodal": "#1 Shipped (327)",
    "our_jaccard": "#2 Our Jaccard (2,396)",
    "our_jaccard_full": "#3 Our Full (6,403)",
}


def giant_component(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(comps[0]).copy()


def fig1_degree_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, fname) in zip(axes, NETWORK_FILES.items()):
        G = nx.read_graphml(PROC_DIR / fname)
        Gc = giant_component(G)
        degrees = np.array([d for _, d in Gc.degree() if d > 0])

        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        fit.plot_ccdf(ax=ax, color="black", marker="o", linestyle="none",
                       markersize=3, label="empirical")
        fit.power_law.plot_ccdf(ax=ax, color="tab:red", linestyle="--", label="power law")
        fit.truncated_power_law.plot_ccdf(ax=ax, color="tab:blue", linestyle="-",
                                           label="truncated power law")
        fit.lognormal.plot_ccdf(ax=ax, color="tab:green", linestyle=":", label="lognormal")

        ax.set_title(NETWORK_LABELS[key])
        ax.set_xlabel("degree $k$")
        ax.set_ylabel("$P(K \\geq k)$")
        ax.legend(fontsize=8)

    fig.suptitle("Degree distribution: empirical CCDF vs. fitted models", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_degree_distribution.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_degree_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_degree_distribution.{pdf,png}")


def fig2_robustness_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    colors = {"random": "tab:gray", "degree": "tab:red", "betweenness": "tab:blue"}
    labels = {"random": "Random removal", "degree": "Degree-targeted", "betweenness": "Betweenness-targeted"}

    for ax, key in zip(axes, NETWORK_FILES.keys()):
        for strat in ["random", "degree", "betweenness"]:
            curve = np.load(PROC_DIR / f"percolation_{key}_{strat}.npy")
            x = np.linspace(0, 1, len(curve))
            ax.plot(x, curve, color=colors[strat], label=labels[strat], linewidth=1.8)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_title(NETWORK_LABELS[key])
        ax.set_xlabel("fraction of nodes removed")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 1)

    axes[0].set_ylabel("giant component fraction")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Robustness to node removal: random vs. targeted attack", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_robustness_curves.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_robustness_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_robustness_curves.{pdf,png}")


def fig3_network_snapshot():
    G = nx.read_graphml(PROC_DIR / "unimodal.graphml")
    Gc = giant_component(G)

    communities = nx.algorithms.community.louvain_communities(Gc, weight=None, seed=42)
    communities = sorted(communities, key=len, reverse=True)
    node_color_idx = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_color_idx[node] = i

    degree = dict(Gc.degree())
    sizes = np.array([degree[n] for n in Gc.nodes()])
    sizes_scaled = 15 + 220 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
    colors = [node_color_idx[n] for n in Gc.nodes()]

    pos = nx.spring_layout(Gc, seed=42, k=0.25, iterations=100)

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(Gc, pos, ax=ax, alpha=0.08, width=0.5)
    nx.draw_networkx_nodes(Gc, pos, ax=ax, node_size=sizes_scaled, node_color=colors,
                            cmap="tab20", linewidths=0.3, edgecolors="white")

    top_nodes = sorted(degree.items(), key=lambda x: -x[1])[:12]
    labels = {n: n for n, _ in top_nodes}
    nx.draw_networkx_labels(Gc, pos, labels=labels, ax=ax, font_size=7)

    ax.set_title("Model #1 (shipped, 327 characters): node size = degree centrality, "
                  "color = detected community", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_network_snapshot.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_network_snapshot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_network_snapshot.{pdf,png}")


if __name__ == "__main__":
    fig1_degree_distribution()
    fig2_robustness_curves()
    fig3_network_snapshot()
    print("\nAll figures saved to", FIG_DIR)
