"""
strategies.py - rules for choosing which experiment to run next.

This is the core of the repository. Everything else is plumbing.

Shared shape:
    Every strategy takes the same inputs and returns the same thing.
        in : mu (predicted value), sigma (uncertainty), best_so_far
        out: a score per candidate. The engine measures the highest-scoring one.
    Keeping the shape identical is what makes swapping strategies a one-line change.

Term: acquisition function
    The formal name for "the rule that assigns the score". It decides how to
    trade off candidates that look good against candidates we know little about.

Term: exploration vs exploitation
    exploitation = dig deeper where results already look good
    exploration  = check regions we know little about, to reduce uncertainty
    The knob that sets the balance (beta, xi) is chosen by a human, not learned.
"""

import numpy as np
from scipy.stats import norm


def random_search(mu, sigma, best_so_far, rng, **kw):
    """Random selection - the baseline.

    Why this matters:
        "UCB found 23%" means nothing on its own. It only becomes a claim once
        there is a baseline: random found 10%. Always report what you compared
        against.

    Uses no model, so mu and sigma are ignored.
    """
    return rng.random(len(mu))


def greedy(mu, sigma, best_so_far, rng, **kw):
    """Greedy selection - take the highest predicted value.

        score = mu

    The trap:
        The model keeps digging in the neighbourhood it already likes. Chosen
        candidates grow more and more similar, and the search can get stuck on a
        local peak. Like only ever visiting highly-rated restaurants in the one
        neighbourhood you already know.

    The defining feature is that it ignores sigma entirely. That makes it the
    counter-example to "a prediction is not a decision".

    Note: greedy is not always wrong. When good candidates are concentrated in
    one family, exploiting that family is the efficient move - see the README.
    """
    return mu


def ucb(mu, sigma, best_so_far, rng, beta=2.0, **kw):
    """UCB (Upper Confidence Bound) - add uncertainty as a bonus.

        score = mu + beta * sigma

    How to read it:
        The upper end of a confidence interval, i.e. "if this goes well, it could
        reach about here". Candidates are ranked by that optimistic estimate.

    beta is the knob:
        beta = 0      -> identical to greedy (no exploration)
        beta large    -> visits unknown regions more often
        The default 2.0 means "count two sigmas' worth of not-knowing".
        There is no universally correct value; a human sets it.

    Analogy:
        A regular place rated 4.2, versus a new place rated 4.0 from 3 reviews.
        Few reviews means it could genuinely be better, so that uncertainty earns
        a curiosity bonus.
    """
    return mu + beta * sigma


def expected_improvement(mu, sigma, best_so_far, rng, xi=0.01, **kw):
    """EI (Expected Improvement).

        score = how much we expect to beat the current best record, on average

    Difference from UCB:
        UCB ranks by an optimistic estimate. EI anchors on the best value measured
        so far and asks for the expected size of the improvement over it. A
        candidate unlikely to beat the record scores near zero.

    Formula:
        z  = (mu - best - xi) / sigma
        EI = (mu - best - xi) * Phi(z) + sigma * phi(z)
        Phi = standard normal CDF, phi = standard normal PDF

        first term  = "how much it beats the record" x "probability it does"
        second term = bonus for being uncertain enough to overshoot
        Because it is a sum of both, high-mean and high-sigma candidates both
        score.

    xi is the knob:
        xi = 0 counts any sliver of improvement, so exploration is weak.
        Larger xi demands a bigger margin before calling it an improvement, which
        pushes the search outward.

    Careful:
        best_so_far is the best value actually measured, not the maximum of the
        model's predictions. Mixing those up breaks EI quietly.
    """
    gap = mu - best_so_far - xi
    z = gap / sigma
    return gap * norm.cdf(z) + sigma * norm.pdf(z)


STRATEGIES = {
    "random": random_search,
    "greedy": greedy,
    "ucb": ucb,
    "ei": expected_improvement,
}


def get_strategy(name):
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(STRATEGIES)}")
    return STRATEGIES[name]
