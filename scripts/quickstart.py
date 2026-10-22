"""
quickstart.py - a 30-second check that everything is installed and working.

Run this first:
    python scripts/quickstart.py

It runs a small version of the experiment (one dataset, short budget, 3 seeds)
and prints a table. The full runs in run_all.py take several minutes each.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdl_engine import adapters, run_many, summarize  # noqa: E402


def main():
    print("Loading data...")
    ds = adapters.load("crossed_barrel")
    print(f"  {ds.X.shape[0]} candidates, {ds.X.shape[1]} features")
    print(f"  objective: {ds.objective}")
    print(f"  true best value in the table: {ds.y.max():.2f}")
    print()
    print("Running 4 strategies x 3 seeds, 30 experiments each (about 30 seconds)...")

    t0 = time.time()
    results = []
    for strat in ("random", "greedy", "ucb", "ei"):
        results += run_many(ds, strategies=(strat,), seeds=range(3),
                            budget=30, n_init=6)
        print(f"  {strat} done")
    print(f"  finished in {time.time() - t0:.0f}s")
    print()

    print(f"{'strategy':10s} {'best found':>12s} {'% of optimum':>14s} {'top 5% found':>14s}")
    print("-" * 54)
    for row in summarize(results, ds, k_fracs=(0.05,)):
        best = row["final_best_mean"]
        print(f"{row['strategy']:10s} {best:12.2f} {100 * best / ds.y.max():13.1f}% "
              f"{100 * row['top5pct_recall_mean']:13.0f}%")

    print()
    print("If you see a table above, the install works.")
    print("Next: python scripts/run_all.py --datasets crossed_barrel   (a few minutes)")


if __name__ == "__main__":
    main()
