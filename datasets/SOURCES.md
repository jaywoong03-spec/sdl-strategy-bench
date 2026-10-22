# Data sources and licences

All three CSVs are redistributed from the
[Olympus](https://github.com/aspuru-guzik-group/olympus) benchmarking suite
(Aspuru-Guzik group), which is released under the MIT licence. A copy of that
licence is in `LICENSE-olympus.txt`. The files here are unmodified copies of the
`data.csv` in each Olympus dataset directory; all reshaping happens in
`sdl_engine/adapters.py`.

Olympus itself is described in:
Häse, Aldeghi, Hickman, Roch, Christensen, Liles, Hein & Aspuru-Guzik,
*Olympus: a benchmarking framework for noisy optimization and experiment
planning*, Machine Learning: Science and Technology 2, 035021 (2021).

---

## `crossed_barrel.csv`

600 rows, no header: `n, theta, r, t, toughness`

Mechanical toughness of 3D-printed crossed-barrel structures, measured by an
autonomous experimentation platform (BEAR).

> Gongora, Xu, Perry, Okoye, Riley, Reyes, Morgan & Brown.
> *A Bayesian experimental autonomous researcher for mechanical design.*
> Science Advances 6(15):eaaz1708, 2020.

## `buchwald_a.csv`

792 rows, no header: `aryl_halide, additive, base, ligand, yield`

Yields from a high-throughput Buchwald-Hartwig C-N cross-coupling screen.
Olympus case "a" (Y=CH, Z=CH, R=CF3). The four factor columns are SMILES
strings; this repository one-hot encodes them.

> Ahneman, Estrada, Lin, Dreher & Doyle.
> *Predicting reaction performance in C-N cross-coupling using machine learning.*
> Science 360:186-190, 2018.

## `redoxmers.csv`

1,408 rows, no header:
`r1_label, r3_label, r4_label, r5_label, abs_lam_diff, ered, gsol`

Candidate redoxmer molecules for non-aqueous redox flow batteries, defined by
substituents at four sites, with three computed properties. All three objectives
are minimised in the source; this repository normalises each to 0-1, flips the
sign, and sums them with equal weights (see the limitations section of the
README).

Distributed with Olympus; consult the Olympus repository for the upstream
reference before citing the underlying study.

---

## A note on what is not here

The workshop's own slides, notebooks and example data are not redistributed in
this repository. Everything here comes from published, openly licensed sources.
