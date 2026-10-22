"""
make_report.py - merge the run_all.py outputs into figures and tables.

Reads one surrogate's runs at a time, because a GP run and a random-forest run
are separate experiments and should not be averaged together.

Usage:
    python scripts/make_report.py                     # the random-forest runs
    python scripts/make_report.py --surrogate gp      # the GP runs

Outputs (for the default random forest, the plain names; otherwise suffixed):
    results/overview.png              the figure at the top of the README
    results/summary.csv               per-strategy means
    results/pairwise.csv              which gaps survive a paired test
    results/run_config.json           reproduction metadata
    results/surrogate_comparison.csv  RF vs GP, same seeds, where both exist
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sdl_engine import plotting, stats  # noqa: E402


def _suffix(sur):
    """Random forest is the default, so its files keep the unadorned names."""
    return "" if sur == "rf" else f"_{sur}"


def _merge_csv(paths, subset):
    parts = [pd.read_csv(p) for p in paths]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=subset, keep="last")


def compare_surrogates(out, k_tag="top1"):
    """RF vs GP for each dataset and strategy, paired on the shared seeds.

    The two surrogates are run on the same seed list, and for a given seed the
    initial design is identical (it is drawn before any model is fitted). So this
    is a paired comparison of the model, with the luck of initialisation held
    fixed - the cleanest way to ask "did changing the surrogate matter".
    """
    rows = []
    for rf_path in sorted(out.glob("raw_*_rf.npz")):
        key = rf_path.stem[len("raw_"):-len("_rf")]
        gp_path = out / f"raw_{key}_gp.npz"
        if not gp_path.exists():
            continue
        rf, gp = np.load(rf_path), np.load(gp_path)
        for strat in plotting.ORDER:
            col = f"recall_{k_tag}_{strat}"
            if col not in rf or col not in gp:
                continue
            a, b = np.asarray(gp[col], float), np.asarray(rf[col], float)
            if len(a) != len(b):
                continue
            row = {"dataset": key, "strategy": strat,
                   "metric": f"{k_tag}pct_recall"}
            row.update(stats.compare_pair(a, b))
            row["mean_gp"] = row.pop("mean_a")
            row["mean_rf"] = row.pop("mean_b")
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="rf", choices=["rf", "gp"])
    args = ap.parse_args()
    sur, sfx = args.surrogate, _suffix(args.surrogate)

    out = ROOT / "results"
    raws = sorted(out.glob(f"raw_*_{sur}.npz"))
    if not raws:
        sys.exit(f"No results/raw_*_{sur}.npz found. "
                 f"Run scripts/run_all.py --surrogate {sur} first.")

    payloads = {p.stem[len("raw_"):-len(f"_{sur}")]: dict(np.load(p)) for p in raws}
    n_seeds = len(next(iter(payloads.values()))["recall_ei"])

    plotting.plot_overview_from_npz(payloads, out / f"overview{sfx}.png",
                                    n_seeds=n_seeds, surrogate=sur)

    summ = _merge_csv(sorted(out.glob(f"summary_*_{sur}.csv")),
                      ["dataset", "strategy"])
    order = {s: i for i, s in enumerate(plotting.ORDER)}
    summ = summ.sort_values(
        ["dataset", "strategy"],
        key=lambda s: s.map(order) if s.name == "strategy" else s)
    summ.to_csv(out / f"summary{sfx}.csv", index=False)

    pair = _merge_csv(sorted(out.glob(f"pairwise_*_{sur}.csv")),
                      ["dataset", "k_frac", "exclude_init",
                       "strategy_a", "strategy_b"])
    if pair is not None:
        pair.to_csv(out / f"pairwise{sfx}.csv", index=False)

    cfgs = {}
    for p in sorted(out.glob(f"run_config_*_{sur}.json")):
        cfgs.update(json.loads(p.read_text()))
    (out / f"run_config{sfx}.json").write_text(
        json.dumps(cfgs, indent=2, ensure_ascii=False))

    cmp_df = compare_surrogates(out)
    if cmp_df is not None:
        cmp_df.to_csv(out / "surrogate_comparison.csv", index=False)

    print(f"Saved: {out}/overview{sfx}.png, {out}/summary{sfx}.csv, "
          f"{out}/pairwise{sfx}.csv, {out}/run_config{sfx}.json"
          + (", surrogate_comparison.csv" if cmp_df is not None else ""))

    cols = [c for c in ["dataset", "strategy", "final_best_mean",
                        "top1pct_recall_mean", "top1pct_recall_sem",
                        "top1pct_recall_postinit_mean", "top5pct_recall_mean"]
            if c in summ.columns]
    print("\n--- means ---")
    print(summ[cols].to_string(index=False))

    if pair is not None:
        head = pair[(pair.k_frac == 0.01) & (~pair.exclude_init)]
        print("\n--- which gaps are real (top 1%, paired across seeds) ---")
        show = head[["dataset", "strategy_a", "strategy_b", "mean_diff",
                     "ci_low", "ci_high", "p_paired_t", "significant"]].copy()
        for c in ("mean_diff", "ci_low", "ci_high"):
            show[c] = (show[c] * 100).round(1)
        print(show.to_string(index=False))

    if cmp_df is not None:
        print("\n--- GP vs RF, same seeds ---")
        show = cmp_df[["dataset", "strategy", "mean_gp", "mean_rf", "mean_diff",
                       "ci_low", "ci_high", "p_paired_t", "significant"]].copy()
        for c in ("mean_gp", "mean_rf", "mean_diff", "ci_low", "ci_high"):
            show[c] = (show[c] * 100).round(1)
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
