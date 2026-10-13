"""Diagnostic: size of the naive vs selection-aware KS test when xmin is chosen
from the data by the CSN rule.

Also reports the max and upper-tail Hill tests at the selected threshold.

Synthetic data mimic one tail side of a return series: n_days = 5000, of which
2000 positive values; the top 300 are an exact power law (zeta = 3) above
x0 = 1 and the rest are uniform on (0, 1).  Under the power-law null a valid
test rejects 5% of the time at the 5% level.
Run: python scripts/10_check_selection_aware.py   (~5 min)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailsurr import constrained_surrogates, fit_mle, surrogate_pvalue, top_hill  # noqa: E402
from tailsurr.fitting import ks_distance  # noqa: E402
from tailsurr.selection import select_tail, selection_aware_ks  # noqa: E402


def one_dataset(rng, n_days=5000, n_pos=2000, n_tail=300, zeta=3.0, n_surr=99):
    tail = rng.random(n_tail) ** (-1 / zeta)
    body = rng.random(n_pos - n_tail)
    values = np.concatenate([tail, body])
    # naive: select xmin, then test with xmin treated as fixed
    k, xmin, t = select_tail(values, n_days)
    obs = ks_distance(t, "powerlaw", fit_mle(t, "powerlaw", xmin), xmin)
    s = constrained_surrogates(t, "powerlaw", xmin, n_surr, rng=rng)
    p_naive = float(surrogate_pvalue(obs, ks_distance(s, "powerlaw", fit_mle(t, "powerlaw", xmin), xmin),
                                     "greater"))
    p_max = float(surrogate_pvalue(t.max(), s.max(-1)))
    p_hill = float(surrogate_pvalue(top_hill(t), top_hill(s)))
    _, p_sel, _ = selection_aware_ks(values, n_days, "powerlaw", n_surr, rng)
    return k, p_naive, p_sel, p_max, p_hill


def main(reps=300):
    rng = np.random.default_rng(3)
    rows = [one_dataset(rng) for _ in range(reps)]
    df = pd.DataFrame(rows, columns=["k_selected", "p_naive", "p_selection_aware",
                                     "p_max", "p_top_hill"])
    df.to_csv(ROOT / "results" / "selection_aware_check.csv", index=False)
    print(f"median selected tail size {df.k_selected.median():.0f} (true power-law part: 300)")
    for col in ["p_naive", "p_selection_aware", "p_max", "p_top_hill"]:
        print(f"{col:18s} rejection at 5%: {np.mean(df[col] <= 0.05):.3f}   "
              f"at 10%: {np.mean(df[col] <= 0.10):.3f}   mean p: {df[col].mean():.3f}")


if __name__ == "__main__":
    main()
