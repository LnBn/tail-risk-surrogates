"""Step 6 (optional): compare our samplers with the authors' reference code.

The authors' code (github.com/JackMurdochMoore/surrogates-for-distributions,
commit 14cb9e0) is used only as an external check, never imported by the
pipeline.  Clone it anywhere and pass the path to its module:

    git clone https://github.com/JackMurdochMoore/surrogates-for-distributions /tmp/ref
    python scripts/06_crosscheck_reference.py /tmp/ref/src/constrained_likelihood_surrogates.py

For each family it draws one input sample (N=60), generates 400 surrogates
with each implementation and compares surrogate statistics (two-sample KS).
It also checks the exponential case against the exact conditional law
(analytic UMVUE), with the reference cut-off passed both as-is and shifted.
"""

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailsurr import constrained_surrogates, sample_family, top_hill, umvue_sf  # noqa: E402

REF_NAMES = {"powerlaw": "powerlaw", "exponential": "expon", "lognormal": "lognorm"}
PARAMS = {"powerlaw": {"alpha": 2.5}, "exponential": {"lam": 1.0},
          "lognormal": {"mu": -1.0, "sigma": 1.0}}


def load_reference(path):
    spec = importlib.util.spec_from_file_location("reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(path, n=60, m=400):
    ref = load_reference(path)
    rng = np.random.default_rng(7)
    np.random.seed(7)
    random.seed(7)  # the reference code uses the global RNGs
    stats_fns = {"max": lambda a: a.max(-1), "top_hill": top_hill,
                 "median": lambda a: np.median(a, -1)}
    print("two-sample KS p-values, ours vs reference (high = indistinguishable)")
    for fam, name in REF_NAMES.items():
        x = sample_family(fam, PARAMS[fam], 1.0, n, rng)
        ours = constrained_surrogates(x, fam, 1.0, m, rng=rng)
        theirs = np.array([ref.gen_surrogate(list(x), model=name, lower_cutoff_hat=1.0)
                           for _ in range(m)])
        line = "  ".join(f"{k}: p={stats.ks_2samp(f(ours), f(theirs)).pvalue:.3f}"
                         for k, f in stats_fns.items())
        print(f"  {fam:12s} {line}")

    # exponential: which implementation matches the exact conditional law?
    x = sample_family("exponential", PARAMS["exponential"], 1.0, n, rng)
    ours = constrained_surrogates(x, "exponential", 1.0, m, rng=rng)
    as_is = np.array([ref.gen_surrogate(list(x), model="expon", lower_cutoff_hat=1.0)
                      for _ in range(m)])
    shifted = np.array([ref.gen_const_exp_surr_merge_2(list(x), lower_cutoff_hat=1.0,
                                                       centr_cutoff_hat=0.0)
                        for _ in range(m)])
    v = np.array([1.5, 2.5, 4.0])
    print("\nexponential, P(X > v) for v =", v)
    print("  exact (UMVUE)        ", umvue_sf(v, x, "exponential", 1.0).round(3))
    for label, s in [("ours", ours), ("reference as-is", as_is),
                     ("reference, cut-off 0", shifted)]:
        print(f"  {label:21s}", (s[:, :, None] > v).mean((0, 1)).round(3))
    print(f"  #(x > 2 xmin): data {np.sum(x > 2)}, reference as-is surrogates "
          f"{sorted(set((as_is > 2).sum(1).tolist()))}, ours range "
          f"{(ours > 2).sum(1).min()}-{(ours > 2).sum(1).max()}")


if __name__ == "__main__":
    main(sys.argv[1])
