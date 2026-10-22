"""
campaign.py - the loop itself.

Term: campaign
    One run of "find the best candidate within a budget of N experiments".
    This is the word the self-driving-lab literature uses.

One campaign (same order as a DMTA cycle):
    0) initial design: measure a few spread-out points (solves the cold start)
    1) Analyze : fit the surrogate on what has been measured
    2) Design  : score the unmeasured candidates, pick the highest
    3) Make/Test: reveal that candidate's value from the oracle table
    4) back to 1 until the budget runs out

What a human decides, not the code:
    - candidate space and ranges      -> adapters.py
    - budget and initial design size  -> arguments below
    - exploration knob (beta, xi)     -> arguments below
    - when to stop                    -> simplified here to "budget exhausted".
      In reality the optimiser does not know the maximum, so a person has to
      judge when to stop.
"""

from dataclasses import dataclass

import numpy as np

from .strategies import get_strategy
from .surrogate import build_surrogate


@dataclass
class CampaignResult:
    """The record of one campaign."""

    dataset: str
    strategy: str
    seed: int
    chosen: np.ndarray       # candidate indices, in the order they were picked
    observed: np.ndarray     # the value revealed for each
    best_curve: np.ndarray   # best-so-far after each experiment
    n_init: int

    def top_k_recall(self, y_all, k_frac=0.01, exclude_init=False):
        """Fraction of the true top k% that the campaign actually picked.

        Why this metric:
            The single best value cannot distinguish "got lucky once" from
            "located the good region". Real screening usually hands several top
            candidates to the next stage, so recall is closer to the actual use.

        exclude_init:
            False counts the whole campaign, including the random initial design.
            True counts only the candidates the strategy chose. The initial design
            is identical across strategies for a given seed, so either version
            compares them fairly - but the default version credits a strategy for
            top candidates that random initialisation handed it for free. The
            `random` arm is the clearest case: with exclude_init=True its score is
            what blind picking actually achieves.
            Both are reported, because the gap between them is itself
            informative: if it is large, the budget is barely doing any work.

        Choosing k_frac:
            Recall can only take n_top + 1 values, so on a small top set the
            metric is coarse - see stats.metric_resolution. k_frac=0.05 is
            reported alongside 0.01 for that reason.
        """
        n_top = max(1, int(round(len(y_all) * k_frac)))
        true_top = set(np.argsort(-y_all)[:n_top].tolist())
        picked = self.chosen[self.n_init:] if exclude_init else self.chosen
        found = len(true_top & set(np.asarray(picked).tolist()))
        return found / len(true_top)


def run_campaign(dataset, strategy="ei", budget=60, n_init=8, seed=0,
                 surrogate="rf", refit_every=1, **strategy_kw):
    """Run one oracle-replay campaign.

    dataset     : adapters.Dataset
    strategy    : 'random' | 'greedy' | 'ucb' | 'ei'
    budget      : total experiments, including the initial design
    n_init      : how many random points to measure up front
    seed        : RNG seed. Results vary a lot by seed, so always run several.
    surrogate   : 'rf' | 'gp'
    refit_every : refit the model every N steps (1 = every step)
    """
    X, y = dataset.X, dataset.y
    n = len(y)
    if budget > n:
        raise ValueError(f"budget ({budget}) exceeds candidate count ({n}).")
    if n_init >= budget:
        raise ValueError(f"n_init ({n_init}) must be smaller than budget ({budget}).")

    rng = np.random.default_rng(seed)
    acq = get_strategy(strategy)

    # --- 0) initial design: spread out, not "where it looks good" ------------
    # Clustering the initial points near a promising spot is how campaigns end up
    # trapped on a local peak.
    measured = list(rng.choice(n, size=n_init, replace=False))
    model = None

    # --- 1..4) iterate -------------------------------------------------------
    for step in range(budget - n_init):
        remaining = np.setdiff1d(np.arange(n), measured)
        best_so_far = y[measured].max()

        if strategy == "random":
            # No model needed on this path; fitting one would only cost time.
            scores = acq(np.zeros(len(remaining)), np.zeros(len(remaining)),
                         best_so_far, rng)
        else:
            if model is None or step % refit_every == 0:
                model = build_surrogate(surrogate, random_state=seed)
                model.fit(X[measured], y[measured])
            mu, sigma = model.predict(X[remaining])
            scores = acq(mu, sigma, best_so_far, rng, **strategy_kw)

        # Break ties at random, otherwise low indices are systematically favoured.
        top = np.flatnonzero(scores == scores.max())
        pick = remaining[rng.choice(top)]
        measured.append(int(pick))

    chosen = np.array(measured)
    observed = y[chosen]
    return CampaignResult(
        dataset=dataset.key,
        strategy=strategy,
        seed=seed,
        chosen=chosen,
        observed=observed,
        best_curve=np.maximum.accumulate(observed),
        n_init=n_init,
    )


def run_many(dataset, strategies=("random", "greedy", "ucb", "ei"),
             seeds=range(20), budget=60, n_init=8, surrogate="rf",
             progress=None, **strategy_kw):
    """Run every strategy x seed combination.

    Why several seeds:
        A single run is dominated by the luck of the initial design. The ranking
        of strategies is far more trustworthy than any single number, so the
        spread across seeds is reported alongside the mean.
    """
    results = []
    seeds = list(seeds)
    total = len(strategies) * len(seeds)
    done = 0
    for strat in strategies:
        for seed in seeds:
            results.append(run_campaign(
                dataset, strategy=strat, budget=budget, n_init=n_init,
                seed=seed, surrogate=surrogate, **strategy_kw))
            done += 1
            if progress:
                progress(done, total, strat, seed)
    return results


def group_by_strategy(results):
    """{strategy name: [CampaignResult, ...]}, seed order preserved."""
    out = {}
    for r in results:
        out.setdefault(r.strategy, []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: r.seed)
    return out


def recall_matrix(results, dataset, k_frac=0.01, exclude_init=False):
    """{strategy name: per-seed recall array}, for the paired tests in stats.py.

    Seeds are sorted the same way for every strategy, so column i of each array
    is the same seed. That alignment is what makes the comparison paired.
    """
    return {s: np.array([r.top_k_recall(dataset.y, k_frac, exclude_init) for r in rs])
            for s, rs in group_by_strategy(results).items()}


def summarize(results, dataset, k_fracs=(0.01, 0.05)):
    """Per-strategy summary rows.

    Standard deviations use ddof=1 (the sample convention), and the standard
    error of the mean is reported next to them: with 20 seeds the spread across
    seeds and the uncertainty in the mean differ by a factor of ~4.5, and quoting
    the first as if it were the second is a common way to overstate a result.
    """
    y = dataset.y
    if isinstance(k_fracs, float):
        k_fracs = (k_fracs,)
    grouped = group_by_strategy(results)

    rows = []
    for strat, rs in grouped.items():
        finals = np.array([r.best_curve[-1] for r in rs])
        n = len(rs)
        row = {
            "strategy": strat,
            "n_seeds": n,
            "final_best_mean": float(finals.mean()),
            "final_best_std": float(finals.std(ddof=1)) if n > 1 else 0.0,
        }
        for k in k_fracs:
            tag = f"top{int(round(k * 100))}pct"
            for suffix, excl in (("", False), ("_postinit", True)):
                vals = np.array([r.top_k_recall(y, k, excl) for r in rs])
                sd = float(vals.std(ddof=1)) if n > 1 else 0.0
                row[f"{tag}_recall{suffix}_mean"] = float(vals.mean())
                row[f"{tag}_recall{suffix}_std"] = sd
                row[f"{tag}_recall{suffix}_sem"] = sd / np.sqrt(n) if n > 1 else 0.0
        rows.append(row)
    return rows
