"""Step 5 (diagnostic): is the default number of MCMC steps enough?

Same input datasets, surrogates generated with 1x and 4x the default number of
transitions (N (log2 N + 10)); if the chains have mixed, statistics of the
surrogates should not depend on the multiplier.  The lognormal sampler is the
one at risk because the xmin threshold makes it reject some proposals.

Run: python scripts/05_mixing_check.py   (~2 min)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy.optimize import brentq  # noqa: E402

from tailsurr import constrained_surrogates, default_n_steps, sample_family, top_hill  # noqa: E402
from tailsurr.fitting import sf  # noqa: E402

TRUE = {"powerlaw": {"alpha": 2.5}, "lognormal": {"mu": -1.0, "sigma": 1.0},
        "exponential": {"lam": 1.0}}


def main(n=50, reps=1000, n_surr=50):
    rng = np.random.default_rng(5)
    rows = []
    for fam, params in TRUE.items():
        x = sample_family(fam, params, 1.0, (reps, n), rng)
        # exact true 99.8% quantile (computed once, shared by both arms)
        v = brentq(lambda t: sf(t, fam, params, 1.0) - 0.002, 1.0, 1e9)
        out = {}
        for mult in (1, 4):
            s = constrained_surrogates(x, fam, 1.0, n_surr,
                                       n_steps=mult * default_n_steps(n), rng=rng)
            # per-dataset surrogate means of each statistic
            out[mult] = {"log max": np.log(s.max(-1)).mean(-1),
                         "top-10% Hill": top_hill(s).mean(-1),
                         "frac > true 99.8% quantile": (s > v).mean((-1, -2))}
        for stat in out[1]:
            d = out[4][stat] - out[1][stat]  # paired difference across datasets
            rows.append({"family": fam, "statistic": stat,
                         "mean_1x": out[1][stat].mean(), "mean_4x": out[4][stat].mean(),
                         "paired_diff": d.mean(), "diff_se": d.std(ddof=1) / np.sqrt(reps),
                         "z": d.mean() / (d.std(ddof=1) / np.sqrt(reps))})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "mixing_check.csv", index=False)
    print(df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
