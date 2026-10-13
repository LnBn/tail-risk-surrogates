"""Breadth step 2: tail diagnostics for every asset, both tails, raw and filtered.

For each asset (22) and tail side (left = falls, right = rises):
  * volatility filter: GARCH(1,1)-normal, GJR-GARCH(1,1)-normal and
    GJR-GARCH(1,1)-Student-t are fitted; the lowest-BIC converged one is used;
  * clustering of top-5% moves (waiting-time CV vs 999 shuffles) for the raw
    series and each filter;
  * threshold by the CSN rule on a grid of tail sizes (50 points .. 20% of days);
  * tail index zeta = alpha - 1 with an asymptotic SE and (filtered series)
    a 200-draw bootstrap CI that re-selects the threshold each time;
  * headline tests at the CSN threshold: power law / lognormal / exponential,
    each with max, upper-tail Hill and KS statistics (999 constrained
    surrogates); raw series are included for comparison only;
  * threshold sensitivity (filtered only): tails = top 1/2/5/10/20% of days,
    499 surrogates.

Outputs: results/breadth_*.csv.  Run after 07:
    python scripts/08_breadth_tail_tests.py      (~20-30 min on 6 cores)
"""

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailsurr import (ALTERNATIVE, FAMILIES, constrained_surrogates,  # noqa: E402
                      shuffle_surrogates, surrogate_pvalue, tail_statistics,
                      vuong_test, waiting_time_cv)
from tailsurr.filters import SPECS, best_filter  # noqa: E402
from tailsurr.quality import screen  # noqa: E402
from tailsurr.selection import bootstrap_zeta, select_tail  # noqa: E402
from tailsurr.universe import UNIVERSE, load_level, to_returns  # noqa: E402

RES = ROOT / "results"
SIDES = {"left": -1.0, "right": 1.0}
M_HEAD, M_SENS = 999, 499
SENS_FRACS = [0.01, 0.02, 0.05, 0.10, 0.20]
SENS_MIN_N = 40
N_WORKERS = 6
SEED = 2026


def clustering(series, side_sign, rng, frac=0.05, n_shuffle=999):
    """Waiting-time CV of the top `frac` moves on one side, with shuffle p-value."""
    v = side_sign * series.values
    exceed = v >= np.quantile(v, 1 - frac)
    obs = waiting_time_cv(exceed)[0]
    surr = waiting_time_cv(shuffle_surrogates(exceed, n_shuffle, rng))
    return obs, float(surrogate_pvalue(obs, surr, "greater"))


def test_families(tail, xmin, n_surr, rng):
    """All family x statistic tests on one tail sample."""
    rows = []
    for fam in FAMILIES:
        surr = constrained_surrogates(tail, fam, xmin, n_surr, rng=rng)
        for stat, (obs, s) in tail_statistics(tail, surr, fam, xmin).items():
            rows.append({"family": fam, "statistic": stat, "observed": float(obs),
                         "surrogate_median": float(np.median(s)),
                         "z": float((obs - s.mean()) / s.std()),
                         "pvalue": float(surrogate_pvalue(obs, s, ALTERNATIVE[stat])),
                         "p_greater": float(surrogate_pvalue(obs, s, "greater")),
                         "p_less": float(surrogate_pvalue(obs, s, "less"))})
    return rows


def analyse_asset(args):
    asset, seed = args
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    r, _, _ = screen(to_returns(load_level(asset), asset.kind), asset.kind, asset.name)
    spec, z, fits = best_filter(r)
    raw = r.copy()
    if asset.kind == "diff":
        # yield changes are quoted in whole basis points: add uniform noise on
        # (-0.5, 0.5) bp so the continuous-data methods see no ties
        raw = raw + rng.uniform(-0.5, 0.5, raw.size)

    base = {"asset": asset.name, "class": asset.asset_class}
    out = {"filters": [], "clustering": [], "tail_index": [], "headline": [], "sensitivity": []}
    for s, (zs, bic, ok) in fits.items():
        out["filters"].append({**base, "spec": s, "bic": bic, "converged": ok,
                               "chosen": s == spec})

    for side, sign in SIDES.items():
        # clustering diagnostics
        for label, series in [("raw", raw)] + [(s, fits[s][0]) for s in SPECS]:
            cv, p = clustering(series, sign, rng)
            out["clustering"].append({**base, "series": label, "side": side,
                                      "chosen_filter": label == spec, "cv": cv, "pvalue": p})

        for label, series in [("raw", raw), ("filtered", z)]:
            values, n_days = sign * series.values, series.size
            k, xmin, tail = select_tail(values, n_days)
            zeta = k / np.sum(np.log(tail / xmin))
            row = {**base, "series": label, "side": side, "filter": spec if label == "filtered" else "",
                   "n_days": n_days, "k": k, "xmin": xmin, "tail_frac": k / n_days,
                   "zeta": zeta, "zeta_se": zeta / np.sqrt(k)}
            if label == "filtered":
                boot = bootstrap_zeta(values, n_days, rng=rng)
                row.update(zeta_boot_lo=np.quantile(boot, 0.025),
                           zeta_boot_hi=np.quantile(boot, 0.975))
            for other in ["lognormal", "exponential"]:
                R, p = vuong_test(tail, "powerlaw", other, xmin)
                row[f"vuong_R_pl_vs_{other[:3]}"], row[f"vuong_p_pl_vs_{other[:3]}"] = R, p
            out["tail_index"].append(row)

            key = {**base, "series": label, "side": side, "k": k, "xmin": xmin}
            out["headline"] += [{**key, **t} for t in test_families(tail, xmin, M_HEAD, rng)]

            if label == "filtered":
                desc = np.sort(values)[::-1]
                for frac in SENS_FRACS:
                    kk = int(frac * n_days)
                    if kk < SENS_MIN_N or desc[kk - 1] <= 0:
                        continue
                    t, xm = desc[:kk], desc[kk - 1]
                    skey = {**base, "side": side, "tail_frac": frac, "k": kk, "xmin": xm}
                    out["sensitivity"] += [{**skey, **x} for x in test_families(t, xm, M_SENS, rng)]
    print(f"done {asset.name} (filter {spec})", flush=True)
    return out


def main():
    seeds = np.random.SeedSequence(SEED).generate_state(len(UNIVERSE))
    # longest series first so the pool stays busy
    order = sorted(range(len(UNIVERSE)), key=lambda i: -len(load_level(UNIVERSE[i])))
    tasks = [(UNIVERSE[i], int(seeds[i])) for i in order]
    results = {key: [] for key in ["filters", "clustering", "tail_index", "headline", "sensitivity"]}
    with ProcessPoolExecutor(N_WORKERS) as pool:
        for out in pool.map(analyse_asset, tasks):
            for key in results:
                results[key] += out[key]
    for key, rows in results.items():
        pd.DataFrame(rows).to_csv(RES / f"breadth_{key}.csv", index=False)
    print("written results/breadth_{filters,clustering,tail_index,headline,sensitivity}.csv")


if __name__ == "__main__":
    main()
