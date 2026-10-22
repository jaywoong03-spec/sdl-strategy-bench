"""
adapters.py - turning a dataset into what the engine reads.

To add a new dataset, add one Dataset here. Nothing else in the engine changes.

The engine needs exactly three things:
    X     : numeric features per candidate (n_candidates x n_features)
    y     : the measured value per candidate, always oriented so bigger is better
    names : feature names, for plots and debugging

Why it works this way:
    Every experiment in this repository uses oracle replay. We take a public
    dataset in which everything has already been measured, and hide it. When a
    strategy says "measure this candidate", the engine reveals that row's value.
    That reduces any problem to a single candidate table, so swapping this
    adapter is all it takes to run a new problem.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


@dataclass
class Dataset:
    """One candidate table handed to the engine."""

    key: str
    title: str
    X: np.ndarray
    y: np.ndarray  # always oriented for maximisation
    feature_names: list
    objective: str
    source: str
    notes: str = ""
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"<Dataset {self.key}: {self.X.shape[0]} candidates, "
                f"{self.X.shape[1]} features, objective={self.objective!r}>")


def _one_hot(df, columns):
    """Expand categorical columns into 0/1 columns.

    A model cannot read "ligand A/B/C", so each level gets its own column. Mapping
    them to 1, 2, 3 instead would imply C is three times A, which is meaningless.
    """
    out = pd.get_dummies(df[columns], columns=columns, dtype=float)
    return out.to_numpy(), list(out.columns)


# ---------------------------------------------------------------------------
# 1) Continuous conditions
# ---------------------------------------------------------------------------
def load_crossed_barrel():
    """Maximise the mechanical toughness of a 3D-printed structure.

    Controlled: number of pillars n, twist angle theta, radius r, thickness t
                (all continuous)
    Objective : toughness (higher is better)

    The source paper is itself an autonomous-experimentation study, which makes
    this the closest match to the self-driving-lab framing.
    """
    cols = ["n", "theta", "r", "t", "toughness"]
    df = pd.read_csv(DATA_DIR / "crossed_barrel.csv", header=None, names=cols)
    return Dataset(
        key="crossed_barrel",
        title="3D-printed structure toughness (4 continuous conditions)",
        X=df[["n", "theta", "r", "t"]].to_numpy(float),
        y=df["toughness"].to_numpy(float),
        feature_names=["n", "theta", "r", "t"],
        objective="maximise toughness",
        source=("Gongora et al., 'A Bayesian experimental autonomous researcher "
                "for mechanical design', Science Advances 6(15):eaaz1708, 2020. "
                "Via Olympus (MIT)."),
        notes="600 measured settings on a grid. Continuous, so GP and RF both fit naturally.",
    )


# ---------------------------------------------------------------------------
# 2) Categorical combinations
# ---------------------------------------------------------------------------
def load_buchwald_a():
    """Maximise yield of a Buchwald-Hartwig C-N coupling reaction.

    Controlled: 3 aryl halides x 22 additives x 3 bases x 4 ligands
                (all categorical, 792 combinations)
    Objective : yield (higher is better)

    Not a single continuous variable. Its shape is completely different from
    dataset 1, which is what makes it a real test of whether the engine is
    general rather than tuned to one problem.
    """
    cols = ["aryl_halide", "additive", "base", "ligand", "yield"]
    df = pd.read_csv(DATA_DIR / "buchwald_a.csv", header=None, names=cols)
    X, names = _one_hot(df, ["aryl_halide", "additive", "base", "ligand"])
    return Dataset(
        key="buchwald_a",
        title="Buchwald-Hartwig reaction yield (categorical combinations)",
        X=X,
        y=df["yield"].to_numpy(float),
        feature_names=names,
        objective="maximise yield",
        source=("Ahneman, Estrada, Lin, Dreher & Doyle, 'Predicting reaction "
                "performance in C-N cross-coupling using machine learning', "
                "Science 360:186-190, 2018. Via Olympus (MIT), case a."),
        notes=(f"Components are SMILES strings in the raw file; one-hot encoding "
               f"gives {X.shape[1]} columns. Distance is weakly defined on "
               f"categorical inputs, so the random forest is the safer surrogate."),
        meta={"raw_columns": ["aryl_halide", "additive", "base", "ligand"]},
    )


# ---------------------------------------------------------------------------
# 3) Larger library, several objectives
# ---------------------------------------------------------------------------
def load_redoxmers(w_ered=1.0, w_gsol=1.0, w_lam=1.0):
    """Screen 1,408 candidate molecules for non-aqueous redox flow batteries.

    Controlled: substituents at 4 sites (categorical, 1,408 combinations)
    Objectives: abs_lam_diff, ered, gsol - all "smaller is better" in the source

    Three objectives are collapsed into one score (scalarisation). This is the
    concrete version of the course point that a second objective needs extra
    terms in the scoring formula, not a new algorithm. The weights are a human
    choice and changing them can change the conclusion, which is stated in the
    README rather than hidden.
    """
    cols = ["r1_label", "r3_label", "r4_label", "r5_label",
            "abs_lam_diff", "ered", "gsol"]
    df = pd.read_csv(DATA_DIR / "redoxmers.csv", header=None, names=cols)
    cat = ["r1_label", "r3_label", "r4_label", "r5_label"]
    X, names = _one_hot(df, cat)

    # Normalise each objective to 0..1, then take a weighted sum. All three are
    # minimised in the source, so the sign is flipped to make bigger better.
    def norm(s):
        s = s.to_numpy(float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else np.zeros_like(s)

    score = -(w_lam * norm(df["abs_lam_diff"])
              + w_ered * norm(df["ered"])
              + w_gsol * norm(df["gsol"]))

    return Dataset(
        key="redoxmers",
        title="Redoxmer library screening (3 objectives combined)",
        X=X,
        y=score,
        feature_names=names,
        objective=f"maximise weighted score (w_lam={w_lam}, w_ered={w_ered}, w_gsol={w_gsol})",
        source="Redoxmers dataset via Olympus (MIT): candidates for non-aqueous redox flow batteries.",
        notes=("Largest of the three at 1,408 candidates. Three objectives were "
               "collapsed into a weighted sum; the weights are a human choice, and "
               "a sensitivity analysis over them is listed as future work."),
        meta={"weights": {"lam": w_lam, "ered": w_ered, "gsol": w_gsol},
              "raw_objectives": ["abs_lam_diff", "ered", "gsol"]},
    )


REGISTRY = {
    "crossed_barrel": load_crossed_barrel,
    "buchwald_a": load_buchwald_a,
    "redoxmers": load_redoxmers,
}


def load(key, **kwargs):
    """Load a dataset by name, e.g. load('buchwald_a')."""
    if key not in REGISTRY:
        raise KeyError(f"Unknown dataset: {key}. Available: {list(REGISTRY)}")
    return REGISTRY[key](**kwargs)


def load_all():
    return [load(k) for k in REGISTRY]
