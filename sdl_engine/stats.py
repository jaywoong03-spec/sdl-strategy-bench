"""
stats.py - deciding which differences between strategies are real.

Why this file exists:
    Without it the benchmark can only say "greedy scored 61% and UCB scored 54%".
    That is not yet a claim. With 20 seeds and a spread of 20-30 points, a gap of
    7 points is inside the noise, and reporting it as a win would be wrong.
    This file separates the gaps that survive a test from the gaps that do not.

Term: paired comparison
    Every strategy is run on the same list of seeds, and for a given seed they
    all start from the identical initial design (campaign.py draws it before the
    strategy is ever consulted). So the seeds are matched pairs, like weighing
    the same people before and after rather than weighing two different groups.
    Pairing removes the seed-to-seed luck from the comparison, which is why a
    paired test is used here and not an independent-samples one.

Term: confidence interval
    The range of differences compatible with the data. If it contains zero, the
    data cannot tell which strategy is better. The interval, not the p-value, is
    the number to read: it shows both direction and size.

Two tests are reported, deliberately:
    paired t    - assumes the per-seed differences are roughly normal
    Wilcoxon    - ranks only, so it survives the skewed and clumped
                  distributions that the recall metric produces
    They almost always agree. When they disagree, trust neither and say so.
"""

import numpy as np
from scipy import stats

ORDER = ("random", "greedy", "ucb", "ei")


def _wilcoxon_p(diff):
    """Wilcoxon signed-rank p-value, guarded for degenerate input.

    The test is undefined when every paired difference is zero, which happens
    easily here: on a dataset whose true top 1% is only 6 candidates, two
    strategies can tie on every single seed.
    """
    if not np.any(diff != 0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff).pvalue)
    except ValueError:
        return float("nan")


def compare_pair(a_vals, b_vals, alpha=0.05):
    """Compare two strategies across matched seeds.

    a_vals, b_vals : per-seed scores, same length, same seed order
    returns        : dict of effect size, interval, and both p-values
    """
    a = np.asarray(a_vals, float)
    b = np.asarray(b_vals, float)
    if a.shape != b.shape:
        raise ValueError(f"paired comparison needs equal lengths, got {a.shape} and {b.shape}")

    diff = a - b
    n = len(diff)
    mean_diff = float(diff.mean())

    if n < 2 or diff.std() == 0:
        # No variation left to test: either one seed, or an exact tie everywhere.
        lo = hi = mean_diff
        p_t = 1.0
    else:
        sem = stats.sem(diff)
        lo, hi = stats.t.interval(1 - alpha, n - 1, mean_diff, sem)
        p_t = float(stats.ttest_rel(a, b).pvalue)

    p_w = _wilcoxon_p(diff)
    # "Significant" requires both tests to agree, which is the conservative
    # reading. The interval is still the number that should be quoted.
    significant = bool(p_t < alpha and (np.isnan(p_w) or p_w < alpha))

    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": mean_diff,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_paired_t": p_t,
        "p_wilcoxon": p_w,
        "n_pairs": n,
        "significant": significant,
    }


def pairwise_table(values_by_strategy, order=ORDER, alpha=0.05, sort_by_mean=True):
    """Every pair of strategies, ranked best mean first.

    values_by_strategy : {strategy name: per-seed scores}
    returns            : list of row dicts, ready for a DataFrame
    """
    present = [s for s in order if s in values_by_strategy]
    present += [s for s in values_by_strategy if s not in present]
    if sort_by_mean:
        present.sort(key=lambda s: -np.mean(values_by_strategy[s]))

    rows = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            row = {"strategy_a": a, "strategy_b": b}
            row.update(compare_pair(values_by_strategy[a],
                                    values_by_strategy[b], alpha=alpha))
            rows.append(row)
    return rows


def metric_resolution(n_candidates, k_frac, n_seeds=None, values=None):
    """How coarse the recall metric is on this dataset.

    Recall is "how many of the true top k% did the campaign pick", so it can only
    take n_top + 1 values. On 600 candidates at k=1% the top set is 6 candidates,
    which means recall moves in jumps of 16.7 points. Quoting a 5-point
    difference on a scale whose smallest step is 16.7 points is reading a ruler
    past its markings, and this function makes that visible instead of implied.
    """
    n_top = max(1, int(round(n_candidates * k_frac)))
    out = {
        "n_candidates": int(n_candidates),
        "k_frac": float(k_frac),
        "n_top": n_top,
        "step_pp": 100.0 / n_top,
        "possible_values": n_top + 1,
    }
    if values is not None:
        vals = np.asarray(values, float)
        out["distinct_observed"] = int(len(np.unique(np.round(vals, 9))))
        out["n_seeds"] = int(len(vals))
    elif n_seeds is not None:
        out["n_seeds"] = int(n_seeds)
    return out


def format_pairwise(rows, scale=100.0, unit="pp"):
    """Plain-text version of pairwise_table, for stdout and the notebook."""
    lines = [f"{'comparison':28s} {'diff':>9s} {'95% CI':>19s} "
             f"{'p(t)':>8s} {'p(W)':>8s}  verdict"]
    for r in rows:
        name = f"{r['strategy_a']} - {r['strategy_b']}"
        verdict = "real" if r["significant"] else "not distinguishable"
        lines.append(
            f"{name:28s} {r['mean_diff'] * scale:+8.1f}{unit} "
            f"[{r['ci_low'] * scale:+6.1f},{r['ci_high'] * scale:+6.1f}] "
            f"{r['p_paired_t']:8.4f} {r['p_wilcoxon']:8.4f}  {verdict}"
        )
    return "\n".join(lines)
