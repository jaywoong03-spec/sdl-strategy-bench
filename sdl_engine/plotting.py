"""
plotting.py - figures.

Deliberate choices:
    1) Draw a band, not just a line. Results vary by seed, so a lone mean line
       would look cleaner than the method really is. The band is the
       interquartile range (25th to 75th percentile).
    2) Colours differ in lightness as well as hue, so they survive greyscale
       printing and colour-vision deficiency. The legend names each series, so
       colour never carries the meaning alone.
    3) The y axis is rescaled to "% of the dataset optimum". The three datasets
       have different units (toughness, yield, score), and this puts them on one
       comparable axis without changing any ranking.
    4) Individual seeds are drawn as dots over the bars, so a single average
       never hides the spread.
    5) The bars carry a 95% confidence interval on the mean. Bars alone invite
       the reader to treat any visible height difference as a result; the
       interval shows which differences the data can actually support. Where two
       intervals overlap heavily, results/pairwise.csv is the thing to read.
"""

import matplotlib

matplotlib.use("Agg")  # save without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as _st  # noqa: E402

ORDER = ("random", "greedy", "ucb", "ei")

LABELS = {
    "random": ("Random (baseline)", "#9AA3AF"),
    "greedy": ("Greedy (mean only)", "#B4531A"),
    "ucb": ("UCB (mean + uncertainty)", "#2F6FB0"),
    "ei": ("EI (expected improvement)", "#14213D"),
}

TITLES = {
    "crossed_barrel": "Crossed barrel\n4 continuous conditions, 600 candidates",
    "buchwald_a": "Buchwald-Hartwig\n4 categorical factors, 792 candidates",
    "redoxmers": "Redoxmer library\n4 categorical sites, 1408 candidates",
}


def _achievement(curves, y_min, y_max):
    """Rescale the best-so-far curve to "% of the dataset optimum".

    The dataset minimum is 0% and the maximum is 100%. This only makes datasets
    with different units comparable on one axis; it preserves every ranking.
    """
    rng = y_max - y_min
    if rng == 0:
        return curves * 0.0
    return (curves - y_min) / rng * 100.0


# ---------------------------------------------------------------------------
# Array-based core functions (used to redraw from the saved npz results)
# ---------------------------------------------------------------------------
def plot_curves_arrays(curves_by_strategy, y_min, y_max, n_init, title,
                       ax=None, show_legend=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.3))

    for strat in ORDER:
        if strat not in curves_by_strategy:
            continue
        curves = _achievement(curves_by_strategy[strat], y_min, y_max)
        xs = np.arange(1, curves.shape[1] + 1)
        label, color = LABELS[strat]
        ax.fill_between(xs, np.percentile(curves, 25, axis=0),
                        np.percentile(curves, 75, axis=0),
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, np.median(curves, axis=0), color=color, linewidth=2.2,
                label=label)

    ax.axvline(n_init, color="#6B7280", linestyle=":", linewidth=1.2)
    ax.annotate("initial design", xy=(n_init, 3), xytext=(n_init + 1, 3),
                fontsize=8, color="#6B7280")
    ax.set_xlabel("Experiments used")
    ax.set_ylabel("Best found (% of dataset optimum)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 103)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if show_legend:
        ax.legend(fontsize=8, loc="lower right", frameon=False)
    return ax


def plot_recall_arrays(recall_by_strategy, title, k_frac=0.01, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.6))

    present = [s for s in ORDER if s in recall_by_strategy]
    rng = np.random.default_rng(0)
    for i, strat in enumerate(present):
        vals = np.asarray(recall_by_strategy[strat], float) * 100
        label, color = LABELS[strat]
        # 95% CI on the mean, not the spread across seeds: the question the bar
        # is answering is "where is the mean", and the dots already show spread.
        if len(vals) > 1 and vals.std() > 0:
            half = _st.t.ppf(0.975, len(vals) - 1) * _st.sem(vals)
        else:
            half = 0.0
        ax.bar(i, vals.mean(), color=color, width=0.62)
        ax.errorbar(i, vals.mean(), yerr=half, fmt="none", ecolor="#111827",
                    elinewidth=1.4, capsize=5, capthick=1.4, zorder=5)
        # The mean label sits low inside the bar in white. Placing it above the
        # bar collides with the seed dots and neither stays readable.
        ax.annotate(f"{vals.mean():.0f}%", xy=(i, 2.5),
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color="#FFFFFF", zorder=4)
        # Overlay the actual per-seed values so the mean cannot oversell.
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.17, 0.17, len(vals)),
                   vals, s=12, color="#14213D", alpha=0.45, zorder=3,
                   linewidth=0)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([LABELS[s][0].split(" (")[0] for s in present],
                       fontsize=9)
    ax.set_ylabel(f"% of true top {int(k_frac * 100)}% found")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, axis="y", linewidth=0.6)
    ax.set_ylim(0, max(105, ax.get_ylim()[1]))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def plot_overview_from_npz(payloads, path, k_frac=0.01, n_seeds=None,
                           surrogate=None):
    """The single figure for the top of the README: curves above, recall below.

    payloads: {dataset key: dict-like loaded from np.load(...)}
    surrogate: named in the caption, because the figure is only valid for one
    """
    keys = [k for k in TITLES if k in payloads] or list(payloads)
    n = len(keys)
    fig, axes = plt.subplots(2, n, figsize=(5.6 * n, 8.6), squeeze=False)

    for j, key in enumerate(keys):
        p = payloads[key]
        curves = {s: p[f"curve_{s}"] for s in ORDER if f"curve_{s}" in p}
        recalls = {s: p[f"recall_{s}"] for s in ORDER if f"recall_{s}" in p}
        title = TITLES.get(key, key)
        plot_curves_arrays(curves, float(p["y_min"]), float(p["y_max"]),
                           int(p["n_init"]), title, ax=axes[0, j],
                           show_legend=(j == 0))
        plot_recall_arrays(recalls, title, k_frac=k_frac, ax=axes[1, j])
        if j > 0:
            axes[0, j].set_ylabel("")
            axes[1, j].set_ylabel("")

    seed_note = f"{n_seeds} seeds" if n_seeds else "multiple seeds"
    sur_note = {"rf": "random-forest surrogate",
                "gp": "Gaussian-process surrogate"}.get(surrogate, surrogate)
    sur_note = f"{sur_note} · " if sur_note else ""
    fig.suptitle(
        "One engine, three different experiment-selection problems\n"
        f"Oracle replay on public experimental datasets · {sur_note}{seed_note} · "
        "line = median, band = interquartile range, dots = individual seeds\n"
        "bars = mean, error bars = 95% confidence interval on the mean "
        "(overlapping intervals mean the gap is not resolved)",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Convenience wrappers that take CampaignResult lists (used in the notebook)
# ---------------------------------------------------------------------------
def _group(results):
    out = {}
    for r in results:
        out.setdefault(r.strategy, []).append(r)
    return out


def plot_curves(results, dataset, ax=None, show_legend=True):
    g = _group(results)
    curves = {s: np.vstack([r.best_curve for r in rs]) for s, rs in g.items()}
    return plot_curves_arrays(
        curves, float(dataset.y.min()), float(dataset.y.max()),
        results[0].n_init, TITLES.get(dataset.key, dataset.key),
        ax=ax, show_legend=show_legend)


def plot_recall(results, dataset, ax=None, k_frac=0.01):
    g = _group(results)
    recalls = {s: [r.top_k_recall(dataset.y, k_frac) for r in rs]
               for s, rs in g.items()}
    return plot_recall_arrays(recalls, TITLES.get(dataset.key, dataset.key),
                              k_frac=k_frac, ax=ax)
