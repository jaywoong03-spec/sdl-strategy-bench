"""
run_all.py - run 4 strategies x N seeds and save the results to results/.

Each dataset takes several minutes, so the normal mode is one dataset at a time.
Every run writes, tagged with the surrogate used:
    results/raw_<dataset>_<surrogate>.npz        per-seed curves and recalls
    results/summary_<tag>_<surrogate>.csv        per-strategy means
    results/pairwise_<tag>_<surrogate>.csv       which gaps survive a paired test
    results/run_config_<tag>_<surrogate>.json    everything needed to reproduce
and scripts/make_report.py merges them into the figure and tables afterwards.

The surrogate is part of every filename on purpose: a GP run and a random-forest
run answer different questions, and an earlier version of this script let the
second overwrite the first.

Usage:
    python scripts/run_all.py --datasets crossed_barrel
    python scripts/run_all.py --datasets buchwald_a
    python scripts/run_all.py --datasets redoxmers
    python scripts/make_report.py

    python scripts/run_all.py                                     # all three (slow)
    python scripts/run_all.py --seeds 30                          # more seeds
    python scripts/run_all.py --datasets crossed_barrel \
                             --surrogate gp                       # GP, continuous data
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdl_engine import adapters, recall_matrix, run_many, stats, summarize  # noqa: E402

# Budgets are set per dataset at roughly 7-10% of the candidate count, so that
# every run is genuinely unable to see the whole space.
BUDGETS = {
    "crossed_barrel": dict(budget=50, n_init=8),
    "buchwald_a": dict(budget=60, n_init=10),
    "redoxmers": dict(budget=100, n_init=12),
}

STRATEGIES = ("random", "greedy", "ucb", "ei")

# Both are reported everywhere. 0.01 is the headline number; 0.05 exists because
# the top 1% of 600 candidates is 6 rows, which makes recall move in steps of
# 16.7 points - too coarse to resolve the gaps being discussed.
K_FRACS = (0.01, 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--surrogate", default="rf", choices=["rf", "gp"])
    ap.add_argument("--beta", type=float, default=2.0, help="UCB exploration knob")
    ap.add_argument("--xi", type=float, default=0.01, help="EI exploration knob")
    ap.add_argument("--k-frac", type=float, default=0.01,
                    help="which top fraction is the headline metric")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="dataset keys to run; empty means all")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(__file__).resolve().parent.parent / args.out
    out.mkdir(exist_ok=True)

    k_fracs = tuple(dict.fromkeys((args.k_frac,) + K_FRACS))
    keys = args.datasets or list(adapters.REGISTRY)
    datasets = [adapters.load(k) for k in keys]
    sur = args.surrogate
    rows, pair_rows = [], []

    for ds in datasets:
        cfg = BUDGETS[ds.key]
        print(f"\n=== {ds.key}: {ds.X.shape[0]} candidates, "
              f"{ds.X.shape[1]} features, budget={cfg['budget']}, "
              f"surrogate={sur}")
        t0 = time.time()

        res = []
        for strat in STRATEGIES:
            kw = {}
            if strat == "ucb":
                kw["beta"] = args.beta
            if strat == "ei":
                kw["xi"] = args.xi
            r = run_many(ds, strategies=(strat,), seeds=range(args.seeds),
                         surrogate=sur, **cfg, **kw)
            res.extend(r)
            finals = np.array([x.best_curve[-1] for x in r])
            rec = np.array([x.top_k_recall(ds.y, args.k_frac) for x in r])
            print(f"  {strat:7s} best={finals.mean():8.3f} "
                  f"(sd {finals.std(ddof=1):6.3f})   "
                  f"top{int(round(args.k_frac * 100))}% recall="
                  f"{rec.mean() * 100:5.1f}% (sd {rec.std(ddof=1) * 100:4.1f})",
                  flush=True)

        # --- which of those gaps are actually distinguishable ----------------
        # Seeds are shared across strategies and the initial design is drawn
        # before the strategy is consulted, so these are matched pairs.
        for k in k_fracs:
            for excl in (False, True):
                mat = recall_matrix(res, ds, k_frac=k, exclude_init=excl)
                for row in stats.pairwise_table(mat):
                    row.update(dataset=ds.key, surrogate=sur, k_frac=k,
                               exclude_init=excl)
                    pair_rows.append(row)

        head = recall_matrix(res, ds, k_frac=args.k_frac)
        resolution = stats.metric_resolution(
            ds.X.shape[0], args.k_frac, values=head["ei"])
        print(f"\n  top{int(round(args.k_frac * 100))}% = {resolution['n_top']} "
              f"candidates, so recall moves in steps of "
              f"{resolution['step_pp']:.1f}pp "
              f"({resolution['possible_values']} possible values)")
        print("  " + stats.format_pairwise(
            stats.pairwise_table(head)).replace("\n", "\n  "))

        # Save only what redrawing and retesting need.
        payload = {"n_init": cfg["n_init"], "budget": cfg["budget"],
                   "y_min": ds.y.min(), "y_max": ds.y.max(),
                   "n_candidates": ds.X.shape[0], "surrogate": sur,
                   "k_frac_headline": args.k_frac}
        for strat in STRATEGIES:
            rs = [x for x in res if x.strategy == strat]
            rs.sort(key=lambda x: x.seed)
            payload[f"curve_{strat}"] = np.vstack([x.best_curve for x in rs])
            for k in k_fracs:
                tag = f"top{int(round(k * 100))}"
                payload[f"recall_{tag}_{strat}"] = np.array(
                    [x.top_k_recall(ds.y, k) for x in rs])
                payload[f"recall_{tag}_postinit_{strat}"] = np.array(
                    [x.top_k_recall(ds.y, k, exclude_init=True) for x in rs])
            # Kept under the original key so older figures still redraw.
            payload[f"recall_{strat}"] = payload[
                f"recall_top{int(round(args.k_frac * 100))}_{strat}"]
        np.savez(out / f"raw_{ds.key}_{sur}.npz", **payload)

        for row in summarize(res, ds, k_fracs=k_fracs):
            row["dataset"] = ds.key
            row["surrogate"] = sur
            row["n_candidates"] = int(ds.X.shape[0])
            row["budget"] = cfg["budget"]
            rows.append(row)
        print(f"\n  ({time.time() - t0:.1f}s)", flush=True)

    tag = "_".join(keys) if args.datasets else "all"

    df = pd.DataFrame(rows)
    lead = ["dataset", "surrogate", "n_candidates", "budget", "strategy", "n_seeds"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(out / f"summary_{tag}_{sur}.csv", index=False)

    pdf = pd.DataFrame(pair_rows)
    plead = ["dataset", "surrogate", "k_frac", "exclude_init",
             "strategy_a", "strategy_b"]
    pdf = pdf[plead + [c for c in pdf.columns if c not in plead]]
    pdf.to_csv(out / f"pairwise_{tag}_{sur}.csv", index=False)

    meta = {
        "seeds": args.seeds,
        "surrogate": sur,
        "ucb_beta": args.beta,
        "ei_xi": args.xi,
        "k_frac_headline": args.k_frac,
        "k_fracs_reported": list(k_fracs),
        "recall_includes_initial_design": "both variants reported",
        "budgets": BUDGETS,
        "datasets": {ds.key: {"n_candidates": int(ds.X.shape[0]),
                              "n_features": int(ds.X.shape[1]),
                              "objective": ds.objective,
                              "source": ds.source} for ds in datasets},
    }
    (out / f"run_config_{tag}_{sur}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\nSaved: {out}/summary_{tag}_{sur}.csv, "
          f"{out}/pairwise_{tag}_{sur}.csv, {out}/raw_*_{sur}.npz")
    print("Build the figure with scripts/make_report.py once all runs finish.")

    show = [c for c in ["dataset", "surrogate", "strategy",
                        "top1pct_recall_mean", "top1pct_recall_postinit_mean",
                        "top5pct_recall_mean"] if c in df.columns]
    print("\n" + df[show].to_string(index=False))


if __name__ == "__main__":
    main()
