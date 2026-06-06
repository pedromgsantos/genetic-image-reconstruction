"""analysis_utils.py — GA random-search history helpers.

Covers both per-metric single-analysis (ga_analysis) and cross-metric
comparison (metric_comparison), as well as the full statistical suite
originally in experimental_comparison_rmse (fitness-across-generations,
boxplots, top-3 error-bars, Wilcoxon matrices, network graph, summary table)
— all generalised so they work for any metric / fitness direction.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------
# Maps the substring found in the `solution` class name → (label, minimize)
# Add new metrics here as they appear.

METRIC_META: dict[str, dict] = {
    "MSE":    {"label": "MSE",    "minimize": True},
    "RMSE":   {"label": "RMSE",   "minimize": True},
    "CIELAB": {"label": "CIELAB", "minimize": True},
    # default (no keyword) — treated as generic minimisation problem
    "DEFAULT": {"label": "Fitness", "minimize": True},
}


def extract_metric(solution_value: str) -> str:
    """Return the metric key embedded in a solution class name.

    Examples
    --------
    'TriangleMSESolution'     → 'MSE'
    'TriangleRMSESolution'    → 'RMSE'
    'TriangleCIELABSolution'  → 'CIELAB'
    'TriangleSolution'        → 'DEFAULT'
    """
    for key in METRIC_META:
        if key == "DEFAULT":
            continue
        if key in solution_value:
            return key
    return "DEFAULT"


def filter_by_metric(df: pd.DataFrame, metric: str | None) -> pd.DataFrame:
    """Return only rows whose solution class matches *metric*.

    Pass ``None`` or ``''`` to skip filtering and return df as-is.
    """
    if not metric:
        return df
    mask = df["metric"] == metric
    result = df[mask]
    if result.empty:
        available = df["metric"].unique().tolist()
        raise ValueError(
            f"No rows found for metric='{metric}'. Available: {available}"
        )
    return result.copy()


def metric_minimize(metric: str) -> bool:
    """Return True if lower fitness is better for this metric."""
    return METRIC_META.get(metric, METRIC_META["DEFAULT"])["minimize"]


def metric_label(metric: str) -> str:
    """Human-readable label for the metric."""
    return METRIC_META.get(metric, METRIC_META["DEFAULT"])["label"]


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

PALETTES: dict[str, dict] = {
    "selection": {
        "ranking":               "#2563eb",
        "fitness_proportionate": "#059669",
        "tournament":            "#d97706",
    },
    "crossover": {
        "single_point": "#7c3aed",
        "order_based":  "#db2777",
        "uniform":      "#0891b2",
        "arithmetic":   "#16a34a",
    },
    "mutation": {
        "mixed":         "#f59e0b",
        "gaussian":      "#ef4444",
        "random_reset":  "#8b5cf6",
        "triangle_swap": "#06b6d4",
    },
    "metric": {
        "MSE":     "#2563eb",
        "RMSE":    "#d97706",
        "CIELAB":  "#7c3aed",
        "DEFAULT": "#059669",
    },
}


def _resolve_palette(column: str, unique_vals: list) -> dict:
    if column in PALETTES:
        return PALETTES[column]
    colors = plt.cm.tab10.colors
    return {v: colors[i % 10] for i, v in enumerate(unique_vals)}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load the search-history CSV; derive ``run_id`` and ``metric`` columns."""
    df = pd.read_csv(path)

    df["metric"] = df["solution"].astype(str).apply(extract_metric)

    for col in ["solution", "metric", "selection", "crossover", "mutation"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df["run_id"] = (
        df["metric"].astype(str) + "_"
        + df["config_id"].astype(str) + "_"
        + df["seed"].astype(str)
    )
    return df


def available_metrics(df: pd.DataFrame) -> list[str]:
    """List the metric keys present in *df*."""
    return sorted(df["metric"].astype(str).unique().tolist())


def get_final_gen(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run — snapshot at the last recorded generation."""
    return (
        df.sort_values("generation")
        .groupby("run_id", observed=True)
        .last()
        .reset_index()
    )


def get_best_run(
    df: pd.DataFrame,
    minimize: bool = True,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Return the best run(s) by final fitness.

    Parameters
    ----------
    minimize : True → lower is better
    group_by : optional column — returns best run *per group*
    """
    final = get_final_gen(df)
    agg_fn = "idxmin" if minimize else "idxmax"

    if group_by is None:
        idx = getattr(final["best_fitness"], agg_fn)()
        return final.loc[[idx]]

    idx = final.groupby(group_by, observed=True)["best_fitness"].agg(agg_fn)
    return final.loc[idx.values]


def get_run_history(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Full generation history for one run, sorted by generation."""
    result = df[df["run_id"] == run_id].sort_values("generation")
    if result.empty:
        raise ValueError(f"run_id '{run_id}' not found.")
    return result


def leaderboard(
    df: pd.DataFrame,
    top_n: int = 10,
    minimize: bool = True,
    group_by_config: bool = True,
) -> pd.DataFrame:
    """Top N runs or configs.

    group_by_config=True  → aggregate seeds, rank by median fitness.
    group_by_config=False → rank individual runs.
    """
    final = get_final_gen(df)
    asc   = minimize

    if group_by_config:
        return (
            final.groupby("config_id", observed=True)
            .agg(
                median_fitness=("best_fitness", "median"),
                mean_fitness=  ("best_fitness", "mean"),
                std_fitness=   ("best_fitness", "std"),
                n_seeds=       ("seed",         "count"),
                selection=     ("selection",    "first"),
                crossover=     ("crossover",    "first"),
                mutation=      ("mutation",     "first"),
                xo_prob=       ("xo_prob",      "first"),
                mut_prob=      ("mut_prob",     "first"),
                n_elites=      ("n_elites",     "first"),
            )
            .sort_values("median_fitness", ascending=asc)
            .head(top_n)
            .reset_index()
        )

    return (
        final.sort_values("best_fitness", ascending=asc)
        [["run_id", "config_id", "seed", "best_fitness",
          "selection", "crossover", "mutation", "xo_prob", "mut_prob"]]
        .head(top_n)
        .reset_index(drop=True)
    )


def dataset_overview(df: pd.DataFrame) -> None:
    """Print a compact summary of the search space."""
    print("=" * 55)
    print(f"  Rows        : {len(df):>10,}")
    print(f"  Unique runs : {df['run_id'].nunique():>10,}")
    print(f"  Config IDs  : {df['config_id'].nunique():>10,}")
    print(f"  Seeds       : {df['seed'].nunique():>10,}")
    print(f"  Generations : 0 – {df['generation'].max()}")
    metrics = df["metric"].value_counts()
    print(f"  Metrics     : {', '.join(metrics.index.astype(str).tolist())}")
    print("=" * 55)
    for col in ["selection", "crossover", "mutation"]:
        vals = df[col].value_counts()
        print(f"\n  {col.upper()} ({len(vals)} unique):")
        for v, n in vals.items():
            print(f"    {v:<25} {n:>8,} rows")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean_curves(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupby([column, "generation"], observed=True)["best_fitness"]
        .mean()
        .reset_index()
    )


def _build_config_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-config_id label table."""
    meta = (
        df.drop_duplicates("config_id")
        .set_index("config_id")[["selection", "crossover", "mutation"]]
    )
    meta["label"] = (
        meta["selection"].astype(str).str.replace("_", " ") + " | "
        + meta["crossover"].astype(str).str.replace("_", " ") + " | "
        + meta["mutation"].astype(str).str.replace("_", " ")
    )
    return meta


def _build_pvalue_matrix(
    final: pd.DataFrame,
    configs: list,
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank p-values across seeds for all config pairs."""
    pivot    = final.pivot_table(index="seed", columns="config_id", values="best_fitness")
    pval_mat = pd.DataFrame(np.nan, index=configs, columns=configs)

    for c1, c2 in combinations(configs, 2):
        x = pivot[c1].dropna()
        y = pivot[c2].dropna()
        common = x.index.intersection(y.index)
        if len(common) < 5:
            continue
        diff = x[common].values - y[common].values
        if np.all(diff == 0):
            pval_mat.loc[c1, c2] = 1.0
            pval_mat.loc[c2, c1] = 1.0
        else:
            _, p = wilcoxon(x[common], y[common])
            pval_mat.loc[c1, c2] = round(p, 4)
            pval_mat.loc[c2, c1] = round(p, 4)

    return pval_mat


# ===========================================================================
# Section A — Fitness across generations
# ===========================================================================


    if save:
        plt.savefig("fig_fitness_across_generations.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_fitness_across_generations(
    df: pd.DataFrame,
    minimize: bool = True,
    fitness_label: str = "Fitness",
    figsize_per_row: tuple[float, float] = (18, 5),
    save: bool = False,
) -> None:
    """Mean & median fitness curves, one subplot pair per selection method.
    Mirrors §1 of experimental_comparison_rmse.
    """
    meta       = _build_config_meta(df)
    df         = df.copy()
    df["label"] = df["config_id"].map(meta["label"])

    selections  = sorted(df["selection"].astype(str).unique())
    config_ids  = sorted(df["config_id"].unique())
    palette     = sns.color_palette("tab10", n_colors=len(config_ids))
    cmap        = {cid: palette[i] for i, cid in enumerate(config_ids)}

    n_rows = len(selections)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(figsize_per_row[0],
                                      figsize_per_row[1] * n_rows))

    def _draw(ax_mean, ax_median, subset, title_suffix):
        for cid, grp in subset.groupby("config_id"):
            stats = grp.groupby("generation")["best_fitness"].agg(["mean", "median"])
            lbl   = meta.loc[cid, "label"]
            c     = cmap[cid]
            ax_mean.plot(stats.index, stats["mean"], color=c, label=lbl, lw=1.2)
            ax_median.plot(stats.index, stats["median"], color=c, label=lbl, lw=1.2)
        for ax, stat in zip([ax_mean, ax_median], ["Mean", "Median"]):
            ax.set_title(f"Fitness Across Generations — {stat} ({title_suffix})")
            ax.set_xlabel("Generation")
            ax.set_ylabel(f"Best Fitness ({fitness_label})")
            if minimize:
                ax.invert_yaxis()
            ax.legend(fontsize=6, loc="upper right", ncol=1)
            ax.grid(alpha=0.3)

    for i, sel in enumerate(selections):
        _draw(axes[i][0], axes[i][1],
              df[df["selection"].astype(str) == sel],
              sel.replace("_", " ").title())

    plt.tight_layout()
    if save:
        plt.savefig("fig_fitness_across_generations.png", dpi=150, bbox_inches="tight")
    plt.show()


# ===========================================================================
# Section B — Boxplots (all configs + per selection)
# ===========================================================================

def plot_boxplots(
    df: pd.DataFrame,
    minimize: bool = True,
    fitness_label: str = "Fitness",
    save: bool = False,
) -> None:
    """Horizontal boxplots of final-generation fitness, sorted by median.
    One overall plot + one per selection method.
    Mirrors §2 of experimental_comparison_rmse.
    """
    meta         = _build_config_meta(df)
    df           = df.copy()
    df["label"]  = df["config_id"].map(meta["label"])
    final        = get_final_gen(df)
    # re-attach label to final gen df
    final["label"] = final["config_id"].map(meta["label"])

    selections   = sorted(final["selection"].astype(str).unique())
    direction    = "lower is better" if minimize else "higher is better"

    order = (
        final.groupby("label")["best_fitness"]
        .median()
        .sort_values(ascending=minimize)
        .index.tolist()
    )

    # All configs
    fig, ax = plt.subplots(figsize=(14, max(8, len(order) * 0.35)))
    sns.boxplot(data=final, x="best_fitness", y="label", order=order,
                palette="Set2", orient="h", ax=ax, width=0.6, fliersize=3)
    ax.set_title(f"Final Generation Fitness per Run — All Configurations")
    ax.set_xlabel(f"Best Fitness ({fitness_label} — {direction})")
    ax.set_ylabel("Configuration")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig("fig_boxplot_all.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Per selection
    for sel in selections:
        sub = final[final["selection"].astype(str) == sel]
        sub_order = (
            sub.groupby("label")["best_fitness"]
            .median()
            .sort_values(ascending=minimize)
            .index.tolist()
        )
        fig, ax = plt.subplots(figsize=(14, max(5, len(sub_order) * 0.4)))
        sns.boxplot(data=sub, x="best_fitness", y="label", order=sub_order,
                    palette="Set2", orient="h", ax=ax, width=0.6, fliersize=3)
        ax.set_title(f"Final Generation Fitness — {sel.replace('_', ' ').title()}")
        ax.set_xlabel(f"Best Fitness ({fitness_label})")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if save:
            plt.savefig(f"fig_boxplot_{sel}.png", dpi=150, bbox_inches="tight")
        plt.show()


# ===========================================================================
# Section C — Top-3 error-bar convergence + Wilcoxon annotation
# ===========================================================================

def plot_top3_convergence(
    df: pd.DataFrame,
    minimize: bool = True,
    fitness_label: str = "Fitness",
    top_n: int = 3,
    error_every: int = 20,
    save: bool = False,
) -> None:
    """Average best-fitness per generation for the top N configs,
    with std error bars and pairwise Wilcoxon annotations.
    Mirrors §2 (second block) of experimental_comparison_rmse.
    """
    meta   = _build_config_meta(df)
    final  = get_final_gen(df)

    top_ids = (
        final.groupby("config_id")["best_fitness"]
        .median()
        .sort_values(ascending=minimize)
        .head(top_n)
        .index.tolist()
    )

    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
    colors     = ["#2166ac", "#d6604d", "#4dac26", "#984ea3", "#ff7f00"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cid in enumerate(top_ids):
        subset     = df[df["config_id"] == cid]
        gen_stats  = subset.groupby("generation")["best_fitness"].agg(["mean", "std"])
        label      = meta.loc[cid, "label"]
        ax.plot(gen_stats.index, gen_stats["mean"],
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                lw=2, label=label)
        ax.errorbar(
            gen_stats.index[::error_every],
            gen_stats["mean"].iloc[::error_every],
            yerr=gen_stats["std"].iloc[::error_every],
            fmt="o", color=colors[i % len(colors)],
            capsize=4, capthick=1.5, elinewidth=1.5, markersize=5,
        )

    # Pairwise Wilcoxon on final gen
    pivot_top = final[final["config_id"].isin(top_ids)].pivot_table(
        index="seed", columns="config_id", values="best_fitness"
    )
    pval_lines = []
    for c1, c2 in combinations(top_ids, 2):
        x = pivot_top[c1].dropna()
        y = pivot_top[c2].dropna()
        common = x.index.intersection(y.index)
        if len(common) >= 5 and not np.all((x[common] - y[common]) == 0):
            _, p = wilcoxon(x[common], y[common])
            sig = "*" if p < 0.05 else "ns"
            pval_lines.append(f"cfg{c1} vs cfg{c2}: p={p:.4f} ({sig})")

    direction = f"lower=better" if minimize else "↑ higher=better"
    ax.set_title(
        f"Average Best Fitness per Generation — Top {top_n} Configurations",
        fontsize=12,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Average Best Fitness ({fitness_label})")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    if pval_lines:
        ax.text(
            0.02, 0.05,
            "\n".join(pval_lines),
            transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    if save:
        plt.savefig("fig_top3_errorbar.png", dpi=150, bbox_inches="tight")
    plt.show()


# ===========================================================================
# Section D — Operator isolated impact
# ===========================================================================

def plot_operator_comparison(
    df: pd.DataFrame,
    minimize: bool = True,
    fitness_label: str = "Fitness",
    save: bool = False,
) -> None:
    """Median convergence curves split by each operator type (selection,
    crossover, mutation) — mirrors §3 of experimental_comparison_rmse.
    """
    operator_cols = ["selection", "crossover", "mutation"]
    titles        = ["Selection Comparison", "Crossover Comparison", "Mutation Comparison"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, col, title in zip(axes, operator_cols, titles):
        unique_vals = sorted(df[col].astype(str).unique())
        palette     = _resolve_palette(col, unique_vals)
        for op in unique_vals:
            subset    = df[df[col].astype(str) == op]
            gen_stats = subset.groupby("generation")["best_fitness"].median()
            ax.plot(gen_stats.index, gen_stats.values,
                    label=op.replace("_", " "),
                    color=palette.get(op, "C0"), lw=2)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(f"Median Best Fitness ({fitness_label})")
        if minimize:
            ax.invert_yaxis()
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Operator Comparison: Median Fitness Over Generations",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    if save:
        plt.savefig("fig_operator_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


# ===========================================================================
# Section E — Wilcoxon heatmaps (all configs + top-N focused)
# ===========================================================================

def plot_wilcoxon_heatmap(
    df: pd.DataFrame,
    alpha: float = 0.05,
    top_n: int | None = None,
    save: bool = False,
) -> pd.DataFrame:
    """Wilcoxon signed-rank p-value heatmap for all configs (or top N).

    Returns the raw p-value matrix so callers can inspect it.
    Mirrors §4-§5 of experimental_comparison_rmse.
    """
    meta   = _build_config_meta(df)
    final  = get_final_gen(df)

    if top_n is not None:
        minimize = metric_minimize(
            df["metric"].astype(str).iloc[0] if "metric" in df.columns else "DEFAULT"
        )
        configs = (
            final.groupby("config_id")["best_fitness"]
            .median()
            .sort_values(ascending=minimize)
            .head(top_n)
            .index.tolist()
        )
        title_suffix = f"Top {top_n} Configurations"
        save_suffix  = f"top{top_n}"
    else:
        configs      = final["config_id"].unique().tolist()
        title_suffix = "All Configurations"
        save_suffix  = "all"

    pval_mat = _build_pvalue_matrix(final, configs)

    if top_n is not None:
        top_labels = {cid: meta.loc[cid, "label"] for cid in configs}
        display_mat = pval_mat.rename(index=top_labels, columns=top_labels)
        annot = True
        fmt   = ".3f"
    else:
        short  = {cid: f"cfg{cid}" for cid in configs}
        display_mat = pval_mat.rename(index=short, columns=short)
        n      = len(configs)
        annot  = n <= 30
        fmt    = ".2f"

    n = len(configs)
    mask = np.triu(np.ones_like(display_mat, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(12, n * 0.5), max(10, n * 0.45)))
    sns.heatmap(
        display_mat.astype(float), mask=mask,
        cmap="RdYlBu", vmin=0, vmax=1,
        annot=annot, fmt=fmt, annot_kws={"size": 6 if n > 20 else 8},
        linewidths=0.3, ax=ax, cbar_kws={"label": "p-value"},
    )
    ax.set_title(
        f"Wilcoxon p-values (α = {alpha}) — {title_suffix}", fontsize=12
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    if save:
        plt.savefig(f"fig_wilcoxon_{save_suffix}.png", dpi=150, bbox_inches="tight")
    plt.show()

    sig_pairs = int((pval_mat < alpha).sum().sum()) // 2
    total     = n * (n - 1) // 2
    print(f"Pairs with p < {alpha} (significantly different): {sig_pairs}")
    print(f"Total pairs: {total}")

    return pval_mat


# ===========================================================================
# Section F — Network graph of statistical dissimilarity
# ===========================================================================

def plot_network_graph(
    df: pd.DataFrame,
    pval_matrix: pd.DataFrame | None = None,
    alpha: float = 0.05,
    save: bool = False,
) -> None:
    """Spring-layout graph where edges connect configs NOT significantly
    different (p ≥ alpha).  Nodes are coloured by selection method.
    Mirrors §6 of experimental_comparison_rmse.

    Parameters
    ----------
    pval_matrix : pre-computed p-value matrix from plot_wilcoxon_heatmap.
                  If None, it is computed here (slower for large datasets).
    """
    meta   = _build_config_meta(df)
    final  = get_final_gen(df)
    configs = final["config_id"].unique().tolist()

    if pval_matrix is None:
        pval_matrix = _build_pvalue_matrix(final, configs)

    G = nx.Graph()
    G.add_nodes_from(configs)
    for c1, c2 in combinations(configs, 2):
        p = pval_matrix.loc[c1, c2]
        if not np.isnan(p) and p >= alpha:
            G.add_edge(c1, c2, weight=p)

    sel_vals   = meta.loc[configs, "selection"].astype(str).unique()
    sel_colors = dict(zip(sorted(sel_vals), sns.color_palette("Set1", len(sel_vals))))
    node_colors = [sel_colors[str(meta.loc[n, "selection"])] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=250, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos,
                            labels={n: f"cfg{n}" for n in G.nodes()},
                            font_size=6, ax=ax)

    legend_patches = [
        mpatches.Patch(color=c, label=s.replace("_", " "))
        for s, c in sel_colors.items()
    ]
    ax.legend(handles=legend_patches, title="Selection Method",
              loc="upper left", fontsize=9)
    ax.set_title(
        f"Statistical Distance Between Configurations\n"
        f"(Node Distance ≈ Dissimilarity — edges = p ≥ {alpha})"
    )
    ax.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig("fig_network_graph.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Nodes: {G.number_of_nodes()} | "
          f"Edges (not significantly different): {G.number_of_edges()}")


# ===========================================================================
# Section G — Summary statistics table
# ===========================================================================

def summary_table(
    df: pd.DataFrame,
    minimize: bool = True,
    fitness_label: str = "Fitness",
    save_csv: str | None = "random_search_summary.csv",
) -> pd.DataFrame:
    """All configurations sorted by median final fitness.
    Mirrors §7 of experimental_comparison_rmse.
    """
    final = get_final_gen(df)
    group_cols = ["config_id", "selection", "crossover", "mutation"]
    # keep only group cols that exist
    group_cols = [c for c in group_cols if c in final.columns]

    summary = (
        final.groupby(group_cols, observed=True)["best_fitness"]
        .agg(mean="mean", median="median", std="std", min="min", max="max")
        .round(4)
        .sort_values("median", ascending=minimize)
        .reset_index()
    )
    direction = "lower is better" if minimize else "higher is better"
    print(f"All configurations sorted by median final fitness ({fitness_label} — {direction}):")
    print(summary.to_string(index=False))
    if save_csv:
        summary.to_csv(save_csv, index=False)
        print(f"\nSaved to {save_csv}")
    return summary


# ===========================================================================
# Legacy / ga_analysis.ipynb — column analysis, run summary, distributions
# ===========================================================================

def plot_column_analysis(
    df: pd.DataFrame,
    column: str,
    minimize: bool = True,
    alpha_individual: float = 0.06,
    show_individual: bool = True,
    max_individual_per_group: int = 40,
    figsize: tuple = (14, 10),
) -> None:
    """2×2 analysis for any categorical column.

    [0,0] row counts per value
    [0,1] avg final fitness per value
    [1,0] mean convergence curve per value
    [1,1] all individual runs + mean (coloured by value)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in DataFrame.")

    unique_vals = df[column].dropna().unique().tolist()
    palette     = _resolve_palette(column, unique_vals)

    final     = get_final_gen(df)
    counts    = df[column].value_counts()
    avg_final = final.groupby(column, observed=True)["best_fitness"].mean().sort_values(
        ascending=minimize
    )
    curves = _mean_curves(df, column)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"GA Analysis — '{column}'", fontsize=13, fontweight="bold")

    # [0,0] counts
    ax = axes[0, 0]
    bars = ax.bar(counts.index, counts.values,
                  color=[palette.get(v, "C0") for v in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Row count per {column}")
    ax.set_xlabel(column); ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", rotation=15)

    # [0,1] avg final fitness
    ax = axes[0, 1]
    bars = ax.bar(avg_final.index, avg_final.values,
                  color=[palette.get(v, "C0") for v in avg_final.index])
    for bar, val in zip(bars, avg_final.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + avg_final.max() * 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    direction = "lower is better" if minimize else "↑ higher is better"
    ax.set_title(f"Avg final fitness per {column}  ({direction})")
    ax.set_xlabel(column); ax.set_ylabel("Avg best_fitness")
    ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", rotation=15)

    # [1,0] mean convergence
    ax = axes[1, 0]
    for val in unique_vals:
        sub = curves[curves[column] == val]
        ax.plot(sub["generation"], sub["best_fitness"],
                color=palette.get(val, "C0"), linewidth=2.2, label=str(val))
    ax.set_title(f"Mean convergence per {column}")
    ax.set_xlabel("Generation"); ax.set_ylabel("Mean best_fitness")
    ax.grid(True, alpha=0.3); ax.legend(title=column)

    # [1,1] individual + mean
    ax = axes[1, 1]
    if show_individual:
        rng = np.random.default_rng(42)
        for val in unique_vals:
            sub_df  = df[df[column] == val]
            run_ids = sub_df["run_id"].unique()
            if len(run_ids) > max_individual_per_group:
                run_ids = rng.choice(run_ids, max_individual_per_group, replace=False)
            for rid in run_ids:
                h = sub_df[sub_df["run_id"] == rid].sort_values("generation")
                ax.plot(h["generation"], h["best_fitness"],
                        color=palette.get(val, "C0"),
                        alpha=alpha_individual, linewidth=0.7)
        for val in unique_vals:
            sub = curves[curves[column] == val]
            ax.plot(sub["generation"], sub["best_fitness"],
                    color=palette.get(val, "C0"), linewidth=2.5,
                    label=f"{val} (mean)")
    ax.set_title(f"All runs + mean — by {column}")
    ax.set_xlabel("Generation"); ax.set_ylabel("best_fitness")
    ax.grid(True, alpha=0.3); ax.legend(title=column, fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_run_summary(
    df: pd.DataFrame,
    run_id: str,
    group_by: str = "selection",
    minimize: bool = True,
) -> None:
    """2×2 dashboard for a single run.

    [0,0] this run's convergence
    [0,1] mean curves per group_by value (same config_id)
    [1,0] all sibling runs + mean
    [1,1] hyperparameter card
    """
    palette     = _resolve_palette(group_by, [])
    history     = get_run_history(df, run_id)
    gens        = history["generation"].values
    fit_hist    = history["best_fitness"].values
    config_id   = history["config_id"].iloc[0]
    meta_row    = history.iloc[0]

    siblings    = df[df["config_id"] == config_id]
    curves      = _mean_curves(siblings, group_by)
    unique_vals = curves[group_by].unique()

    meta_text = (
        f"config_id : {meta_row['config_id']}\n"
        f"seed      : {meta_row['seed']}\n"
        f"metric    : {meta_row['metric']}\n"
        f"selection : {meta_row['selection']}\n"
        f"crossover : {meta_row['crossover']}\n"
        f"mutation  : {meta_row['mutation']}\n"
        f"xo_prob   : {meta_row['xo_prob']}\n"
        f"mut_prob  : {meta_row['mut_prob']}\n"
        f"n_elites  : {meta_row['n_elites']}\n"
        f"sigma     : {meta_row['sigma_start']} → {meta_row['sigma_end']}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Run Summary  ·  {run_id}", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(gens, fit_hist, linewidth=1.8)
    ax.set_title("GA Convergence (this run)")
    ax.set_xlabel("Generation"); ax.set_ylabel("Best fitness")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for val in unique_vals:
        sub = curves[curves[group_by] == val]
        ax.plot(sub["generation"], sub["best_fitness"],
                color=palette.get(val, None) or None,
                linewidth=2.0, label=str(val))
    ax.plot(gens, fit_hist, color="grey", linewidth=1.0,
            linestyle="--", alpha=0.7, label="this run")
    ax.set_title(f"Mean curves by {group_by}  (config {config_id})")
    ax.set_xlabel("Generation"); ax.set_ylabel("Mean best_fitness")
    ax.grid(True, alpha=0.3); ax.legend(title=group_by, fontsize=8)

    ax = axes[1, 0]
    for val in unique_vals:
        sub_df = siblings[siblings[group_by] == val]
        color  = palette.get(val, None) or None
        first  = True
        for rid in sub_df["run_id"].unique():
            h = sub_df[sub_df["run_id"] == rid].sort_values("generation")
            ax.plot(h["generation"], h["best_fitness"],
                    color=color, alpha=0.08, linewidth=0.7,
                    label=str(val) if first else None)
            first = False
    for val in unique_vals:
        sub = curves[curves[group_by] == val]
        ax.plot(sub["generation"], sub["best_fitness"],
                color=palette.get(val, None) or None, linewidth=2.2)
    ax.set_title(f"All sibling runs  (config {config_id})")
    ax.set_xlabel("Generation"); ax.set_ylabel("best_fitness")
    ax.grid(True, alpha=0.3); ax.legend(title=group_by, fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.5, 0.5, meta_text, transform=ax.transAxes,
            fontsize=10, family="monospace",
            verticalalignment="center", horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.7", alpha=0.07))
    direction = "lower=better" if minimize else "↑ higher=better"
    best_val  = fit_hist.min() if minimize else fit_hist.max()
    ax.set_title(f"Hyperparams  ·  best = {best_val:.3f}  ({direction})")

    plt.tight_layout()
    plt.show()


def plot_hyperparam_correlations(df: pd.DataFrame) -> None:
    """Heatmap of Pearson correlations between numeric hyperparams and final fitness."""
    final = get_final_gen(df)
    candidate_cols = ["best_fitness", "n_elites", "tournament_size",
                      "swap_weight", "xo_prob", "mut_prob",
                      "sigma_start", "sigma_end", "plateau_window"]
    cols = [c for c in candidate_cols if c in final.columns]
    corr = final[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=0.4, ax=ax, annot_kws={"size": 8})
    ax.set_title("Pearson Correlation — Hyperparams vs Final Fitness", pad=10)
    plt.tight_layout()
    plt.show()


def plot_fitness_distributions(df: pd.DataFrame) -> None:
    """Box plots of final fitness for each categorical hyperparameter."""
    final = get_final_gen(df)
    cats  = [("selection", PALETTES["selection"]),
             ("crossover", PALETTES["crossover"]),
             ("mutation",  PALETTES["mutation"])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Final Fitness Distribution per Category", fontsize=12, fontweight="bold")

    for ax, (col, pal) in zip(axes, cats):
        order = (
            final.groupby(col, observed=True)["best_fitness"]
            .median()
            .sort_values()
            .index
        )
        sns.boxplot(data=final, x=col, y="best_fitness", order=order,
                    palette=pal,
                    flierprops=dict(marker=".", markersize=3, alpha=0.4), ax=ax)
        ax.set_title(col); ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()


# ===========================================================================
# Cross-metric comparison (metric_comparison.ipynb)
# ===========================================================================

def plot_metric_convergence_comparison(
    df: pd.DataFrame,
    group_by: str = "selection",
) -> None:
    """One subplot per metric — mean convergence by *group_by*."""
    metrics = available_metrics(df)
    n       = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub      = filter_by_metric(df, metric)
        minimize = metric_minimize(metric)
        curves   = _mean_curves(sub, group_by)
        palette  = _resolve_palette(group_by, curves[group_by].unique().tolist())

        for val in curves[group_by].unique():
            c = curves[curves[group_by] == val]
            ax.plot(c["generation"], c["best_fitness"],
                    color=palette.get(val, "C0"), linewidth=2.2, label=str(val))

        direction = "↓" if minimize else "↑"
        ax.set_title(f"{metric_label(metric)}  ({direction})")
        ax.set_xlabel("Generation"); ax.set_ylabel("Mean best_fitness")
        ax.grid(True, alpha=0.3); ax.legend(title=group_by, fontsize=8)

    fig.suptitle(
        f"Mean convergence by metric  ·  grouped by {group_by}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_metric_best_params_comparison(
    df: pd.DataFrame,
    top_n: int = 5,
) -> None:
    """For each metric, top-N configs as a horizontal bar chart coloured by selection."""
    metrics = available_metrics(df)
    n       = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub      = filter_by_metric(df, metric)
        minimize = metric_minimize(metric)
        lb       = leaderboard(sub, top_n=top_n, minimize=minimize, group_by_config=True)
        palette  = _resolve_palette("selection", lb["selection"].unique().tolist())

        colors = [palette.get(s, "C0") for s in lb["selection"]]
        ax.barh(range(len(lb)), lb["median_fitness"], color=colors)
        ax.set_yticks(range(len(lb)))
        ax.set_yticklabels([f"cfg {r}" for r in lb["config_id"]], fontsize=8)
        ax.invert_yaxis()
        direction = "lower=better" if minimize else "↑ higher=better"
        ax.set_title(f"{metric_label(metric)} top {top_n}  ({direction})")
        ax.set_xlabel("Median final fitness")
        ax.grid(axis="x", alpha=0.3)

        for i, (_, row) in enumerate(lb.iterrows()):
            ax.text(lb["median_fitness"].max() * 0.01, i,
                    f"  {row['selection']} / {row['crossover']} / {row['mutation']}",
                    va="center", fontsize=7)

    fig.suptitle(f"Top-{top_n} configs per metric", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_metric_param_distribution(
    df: pd.DataFrame,
    param: str = "selection",
    top_n: int = 10,
) -> None:
    """Bar charts showing parameter distribution in top-N configs per metric."""
    metrics = available_metrics(df)
    n       = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub      = filter_by_metric(df, metric)
        minimize = metric_minimize(metric)
        lb       = leaderboard(sub, top_n=top_n, minimize=minimize, group_by_config=True)
        counts   = lb[param].value_counts()
        palette  = _resolve_palette(param, counts.index.tolist())

        ax.bar(counts.index, counts.values,
               color=[palette.get(v, "C0") for v in counts.index])
        ax.set_title(f"{metric_label(metric)}  — top {top_n}")
        ax.set_xlabel(param); ax.set_ylabel("Count in top configs")
        ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", rotation=15)

    fig.suptitle(
        f"'{param}' distribution in top-{top_n} configs per metric",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def table_best_per_metric(
    df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Tidy DataFrame: best top_n configs for every metric, stacked."""
    metrics = available_metrics(df)
    frames  = []
    for metric in metrics:
        sub      = filter_by_metric(df, metric)
        minimize = metric_minimize(metric)
        lb       = leaderboard(sub, top_n=top_n, minimize=minimize, group_by_config=True)
        lb.insert(0, "metric", metric)
        frames.append(lb)
    return pd.concat(frames, ignore_index=True)