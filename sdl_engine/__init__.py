"""
sdl_engine - a small engine for comparing "what should we measure next"
strategies on public experimental datasets under a fixed experiment budget.

Usage:
    from sdl_engine import adapters, run_many, summarize
    ds = adapters.load("buchwald_a")
    results = run_many(ds, budget=60, seeds=range(20))
    summarize(results, ds)
    stats.pairwise_table(recall_matrix(results, ds))   # which gaps are real

To add a dataset, add one Dataset to adapters.py. Nothing else changes.
"""

from . import adapters, plotting, stats, strategies, surrogate  # noqa: F401
from .campaign import (  # noqa: F401
    CampaignResult, group_by_strategy, recall_matrix, run_campaign, run_many,
    summarize,
)

__all__ = [
    "adapters", "strategies", "surrogate", "plotting", "stats",
    "run_campaign", "run_many", "summarize", "CampaignResult",
    "recall_matrix", "group_by_strategy",
]
