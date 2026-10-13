"""Step 3: pilot on real daily returns - which family describes the loss tail?

For each asset (S&P 500, BTC, EUR/USD) and each series
    raw   : daily losses  L_t = -r_t   (percent log returns)
    garch : GARCH(1,1)-standardised residual losses  -z_t
we
  1. test whether tail exceedances are clustered in time (waiting-time CV,
     shuffle surrogates) - i.e. whether the i.i.d. assumption is plausible;
  2. choose the tail threshold xmin by the Clauset-Shalizi-Newman KS method;
  3. test power law / lognormal / exponential tails with constrained
     surrogates and three tail statistics, plus the standard baselines
     (CSN parametric-bootstrap KS test and Vuong likelihood-ratio tests);
  4. repeat step 3 at fixed tail fractions to measure sensitivity to xmin.

Outputs: results/pilot_*.csv, figures/pilot_*.png
Run:     python scripts/03_pilot_tail_tests.py   (needs data from step 2; ~10 min)
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailsurr import (ALTERNATIVE, FAMILIES, constrained_surrogates, fit_mle,  # noqa: E402
                      select_xmin, shuffle_surrogates, surrogate_pvalue,
                      tail_statistics, vuong_test, waiting_time_cv)
from tailsurr.data import ASSETS, garch_standardised_residuals, load_prices, log_returns  # noqa: E402
from tailsurr.fitting import ks_distance, sample_family, sf  # noqa: E402
from tailsurr.plotting import COLORS, LABELS, plt  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"
SEED = 7
N_SURR = 999
TAIL_FRACS = [0.01, 0.02, 0.05, 0.10, 0.20]  # sensitivity: fraction of all days
N_TAIL_MIN = 40
N_WORKERS = 6


def build_series():
    """{(asset, series): pd.Series of losses} for raw and GARCH-filtered data."""
    out = {}
    for asset in ASSETS:
        r = log_returns(load_prices(asset))
        z, _ = garch_standardised_residuals(r)
        out[(asset, "raw")] = -r
        out[(asset, "garch")] = -z
    return out


def clustering_test(losses, tail_frac=0.05, n_shuffle=999, rng=None):
    """Are the top `tail_frac` losses clustered in time?  Shuffle-surrogate test."""
    exceed = (losses >= losses.quantile(1 - tail_frac)).values
    obs = waiting_time_cv(exceed)[0]
    surr = waiting_time_cv(shuffle_surrogates(exceed, n_shuffle, rng))
    return obs, np.median(surr), float(surrogate_pvalue(obs, surr, "greater"))


def csn_bootstrap_pvalue(tail, xmin, n_boot=999, rng=None):
    """Classic CSN goodness-of-fit p-value for a power law above a *fixed* xmin.

    Parametric bootstrap from the fitted law, refitting alpha on every
    synthetic sample (CSN 2009 also re-select xmin; we hold it fixed so the
    comparison with constrained surrogates is like-for-like).
    """
    rng = np.random.default_rng(rng)
    fit = fit_mle(tail, "powerlaw", xmin)
    obs = ks_distance(tail, "powerlaw", fit, xmin)
    synth = sample_family("powerlaw", fit, xmin, (n_boot, tail.size), rng)
    alphas = 1 + tail.size / np.log(synth / xmin).sum(axis=1)
    xs = np.sort(synth, axis=1)
    f = 1 - (xs / xmin) ** (1 - alphas[:, None])
    i = np.arange(1, tail.size + 1)
    d = np.maximum((i / tail.size - f).max(1), (f - (i - 1) / tail.size).max(1))
    return float(surrogate_pvalue(obs, d, "greater"))


def test_tail(args):
    """All family x statistic surrogate tests for one tail sample."""
    key, tail, xmin, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    for fam in FAMILIES:
        surr = constrained_surrogates(tail, fam, xmin, N_SURR, rng=rng)
        for stat, (obs, s) in tail_statistics(tail, surr, fam, xmin).items():
            rows.append({**key, "family": fam, "statistic": stat,
                         "observed": float(obs), "surrogate_median": float(np.median(s)),
                         "pvalue": float(surrogate_pvalue(obs, s, ALTERNATIVE[stat]))})
    return rows


def plot_ccdfs(series, choices):
    """Empirical CCDF of each tail with the three fitted families."""
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6))
    for c, asset in enumerate(ASSETS):
        for r, kind in enumerate(["raw", "garch"]):
            ax = axes[r, c]
            losses = series[(asset, kind)].values
            xmin = choices[(asset, kind)]["xmin"]
            tail = np.sort(losses[losses >= xmin])
            n_all = np.sum(losses > 0)
            # empirical CCDF of all positive losses, P(L > x | L > 0)
            pos = np.sort(losses[losses > 0])
            ax.loglog(pos, 1 - np.arange(pos.size) / pos.size, ".", ms=2,
                      color="#3d3d3a", label="data (positive losses)")
            grid = np.geomspace(xmin, tail.max() * 1.5, 200)
            scale = tail.size / n_all  # fitted tail laws are conditional on L >= xmin
            for fam in FAMILIES:
                p = fit_mle(tail, fam, xmin)
                ax.loglog(grid, scale * sf(grid, fam, p, xmin), color=COLORS[fam],
                          label=f"{LABELS[fam]} fit")
            ax.axvline(xmin, color=COLORS["reference"], ls=":", lw=1)
            unit = "%" if kind == "raw" else "sd"
            ax.set_xlabel(f"loss ({unit})")
            ax.set_title(f"{asset}, {'raw losses' if kind == 'raw' else 'GARCH residual losses'}"
                         f"\nxmin={xmin:.2f}, n_tail={tail.size}")
            ax.set_ylim(0.5 / n_all, 1.5)
    axes[0, 0].set_ylabel("P(L > x | L > 0)")
    axes[1, 0].set_ylabel("P(L > x | L > 0)")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "pilot_ccdfs.png")
    plt.close(fig)


def plot_sensitivity(sens):
    stats_order = ["max", "top_hill", "ks"]
    titles = {"max": "sample max (two-sided)", "top_hill": "top-10% Hill (two-sided)",
              "ks": "KS to MLE fit (upper)"}
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 8), sharex=True, sharey=True)
    for c, asset in enumerate(ASSETS):
        for r, stat in enumerate(stats_order):
            ax = axes[r, c]
            for fam in FAMILIES:
                for kind, ls, mk in [("garch", "-", "o"), ("raw", "--", "s")]:
                    g = sens[(sens.asset == asset) & (sens.series == kind)
                             & (sens.family == fam) & (sens.statistic == stat)]
                    g = g.sort_values("tail_frac")
                    ax.plot(100 * g.tail_frac, g.pvalue, ls, marker=mk, ms=3.5,
                            color=COLORS[fam], mfc=COLORS[fam] if kind == "garch" else "white",
                            label=f"{LABELS[fam]}, {kind}")
            ax.axhline(0.05, color=COLORS["reference"], ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(5e-4, 1.5)
            ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
            if r == 0:
                ax.set_title(asset)
            if c == 0:
                ax.set_ylabel(f"{titles[stat]}\np-value")
            if r == 2:
                ax.set_xlabel("tail = top x% of days")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Sensitivity of constrained-surrogate p-values to the threshold ({N_SURR} "
                 f"surrogates; smallest attainable p = 0.001 one-sided, 0.002 two-sided)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "pilot_sensitivity.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    series = build_series()

    # 1. clustering of exceedances --------------------------------------------
    clus = []
    for (asset, kind), losses in series.items():
        obs, med, p = clustering_test(losses, rng=rng)
        clus.append({"asset": asset, "series": kind, "cv_waiting_time": obs,
                     "cv_shuffled_median": med, "pvalue": p})
    clus = pd.DataFrame(clus)
    clus.to_csv(RES / "pilot_clustering.csv", index=False)
    print(clus.round(4).to_string(index=False))

    # 2. CSN threshold + baselines --------------------------------------------
    choices, base = {}, []
    for (asset, kind), losses in series.items():
        sel = select_xmin(losses.values, n_tail_min=50)
        choices[(asset, kind)] = sel
        tail = losses.values[losses.values >= sel["xmin"]]
        R_ln, p_ln = vuong_test(tail, "powerlaw", "lognormal", sel["xmin"])
        R_ex, p_ex = vuong_test(tail, "powerlaw", "exponential", sel["xmin"])
        base.append({"asset": asset, "series": kind, "n_obs": losses.size,
                     "xmin": sel["xmin"], "n_tail": tail.size,
                     "tail_frac": tail.size / losses.size,
                     "zeta_hat": sel["alpha"] - 1, "ks": sel["ks"],
                     "csn_bootstrap_p": csn_bootstrap_pvalue(tail, sel["xmin"], rng=rng),
                     "vuong_R_pl_vs_ln": R_ln, "vuong_p_pl_vs_ln": p_ln,
                     "vuong_R_pl_vs_exp": R_ex, "vuong_p_pl_vs_exp": p_ex})
    base = pd.DataFrame(base)
    base.to_csv(RES / "pilot_thresholds_and_baselines.csv", index=False)
    print(base.round(4).to_string(index=False))
    plot_ccdfs(series, choices)

    # 3 + 4. constrained-surrogate tests at the CSN and fixed-fraction thresholds
    tasks = []
    for (asset, kind), losses in series.items():
        thresholds = [("csn", choices[(asset, kind)]["xmin"])]
        thresholds += [(f, losses.quantile(1 - f)) for f in TAIL_FRACS]
        for label, xmin in thresholds:
            tail = losses.values[losses.values >= xmin]
            if tail.size < N_TAIL_MIN:
                continue
            key = {"asset": asset, "series": kind,
                   "threshold": "csn" if label == "csn" else "fixed",
                   "tail_frac": tail.size / losses.size, "xmin": xmin, "n_tail": tail.size}
            tasks.append((key, tail, xmin, int(rng.integers(2**31))))
    with ProcessPoolExecutor(N_WORKERS) as pool:
        rows = [r for res in pool.map(test_tail, tasks) for r in res]
    res = pd.DataFrame(rows)
    res.to_csv(RES / "pilot_surrogate_tests.csv", index=False)

    csn = res[res.threshold == "csn"].pivot_table(
        index=["asset", "series", "n_tail"], columns=["family", "statistic"], values="pvalue")
    print(csn.round(3).to_string())
    plot_sensitivity(res[res.threshold == "fixed"])


def replot():
    """Redraw the sensitivity figure from saved results."""
    res = pd.read_csv(RES / "pilot_surrogate_tests.csv")
    plot_sensitivity(res[res.threshold == "fixed"])


if __name__ == "__main__":
    replot() if "--replot" in sys.argv else main()
