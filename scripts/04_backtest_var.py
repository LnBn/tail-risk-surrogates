"""Step 4: rolling out-of-sample one-day VaR / ES backtest (GARCH + tail model).

Every REFIT days we fit a GARCH(1,1) to the trailing WINDOW of returns, take
the standardised residual losses, and estimate their q-quantile (VaR) and
tail mean (ES) with several tail models.  Between refits the GARCH volatility
is updated daily with the frozen parameters, so every forecast uses only past
data.  Forecast:  VaR_t = -mu + sigma_t * VaR_z   (same for ES).

Methods compared
  HS (raw)          unconditional historical simulation on raw losses
  GARCH-normal      normal residuals
  FHS               filtered historical simulation (empirical residual quantile)
  GPD (POT)         McNeil-Frey: generalised Pareto above the k-th largest
  Pareto MLE        Hill / power-law MLE above the same threshold
  CS-powerlaw / CS-lognormal / CS-exponential
                    constrained surrogates of the k residual tail values

Outputs: results/backtest_<asset>.csv (forecasts), results/backtest_<asset>_summary.csv,
         figures/backtest_<asset>.png
Run:     python scripts/04_backtest_var.py [ASSET]   (default SP500; ~5 min)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailsurr.backtest import (christoffersen, kupiec_pof, surrogate_pool,  # noqa: E402
                               var_es_empirical, var_es_from_pool, var_es_gpd,
                               var_es_normal, var_es_pareto)
from tailsurr.data import load_prices, log_returns  # noqa: E402
from tailsurr.plotting import COLORS, plt  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"
WINDOW = 1000     # trailing estimation window (trading days)
REFIT = 20        # re-estimate GARCH and tail every REFIT days
K_TAIL = 100      # number of residual tail values (top 10% of the window)
LEVELS = (0.99, 0.995)
N_SURR = 200
TEST_START = {"SP500": "2000-01-01", "BTC": "2019-01-01", "EURUSD": "2005-01-01"}
SEED = 11

TAIL_METHODS = ["FHS", "GPD (POT)", "Pareto MLE",
                "CS-powerlaw", "CS-lognormal", "CS-exponential"]


def residual_var_es(y, method, rng):
    """{level: (VaR, ES)} of standardised residual losses y for one tail method."""
    if method == "FHS":
        return {q: var_es_empirical(y, q) for q in LEVELS}
    if method == "GPD (POT)":
        return {q: var_es_gpd(y, q, K_TAIL) for q in LEVELS}
    if method == "Pareto MLE":
        return {q: var_es_pareto(y, q, K_TAIL) for q in LEVELS}
    if method.startswith("CS-"):
        # one set of surrogates per refit, shared by both VaR levels
        pooled = surrogate_pool(y, K_TAIL, method[3:], N_SURR, rng)
        return {q: var_es_from_pool(pooled, q, y.size, K_TAIL) for q in LEVELS}
    raise ValueError(method)


def run_backtest(asset):
    from arch import arch_model

    rng = np.random.default_rng(SEED)
    r = log_returns(load_prices(asset))
    dates, rv = r.index, r.values
    first = max(WINDOW, int(np.searchsorted(dates, pd.Timestamp(TEST_START[asset]))))
    methods = ["HS (raw)", "GARCH-normal"] + TAIL_METHODS
    var = {(m, q): np.full(rv.size, np.nan) for m in methods for q in LEVELS}
    es = {(m, q): np.full(rv.size, np.nan) for m in methods for q in LEVELS}

    for t0 in range(first, rv.size, REFIT):
        window = rv[t0 - WINDOW:t0]
        fit = arch_model(window, mean="Constant", vol="GARCH", p=1, q=1,
                         dist="normal", rescale=False).fit(disp="off")
        mu, omega, alpha, beta = fit.params[["mu", "omega", "alpha[1]", "beta[1]"]]
        y = -(fit.resid / fit.conditional_volatility)  # residual losses
        resid_q = {(m, q): v for m in TAIL_METHODS
                   for q, v in residual_var_es(y, m, rng).items()}
        for q in LEVELS:
            resid_q[("GARCH-normal", q)] = var_es_normal(q)
            resid_q[("HS (raw)", q)] = var_es_empirical(-window, q)  # raw losses

        # one-step-ahead volatility for each day until the next refit
        sigma2 = fit.conditional_volatility[-1] ** 2
        eps = fit.resid[-1]
        for t in range(t0, min(t0 + REFIT, rv.size)):
            sigma2 = omega + alpha * eps ** 2 + beta * sigma2   # uses data up to t-1
            for (m, q), (v_z, e_z) in resid_q.items():
                if m == "HS (raw)":
                    var[(m, q)][t], es[(m, q)][t] = v_z, e_z
                else:
                    var[(m, q)][t] = -mu + np.sqrt(sigma2) * v_z
                    es[(m, q)][t] = -mu + np.sqrt(sigma2) * e_z
            eps = rv[t] - mu  # observe today's return, then move on
        print(f"{asset}: {dates[t0].date()}", end="\r", flush=True)

    sl = slice(first, rv.size)
    loss = -rv[sl]
    forecasts = pd.DataFrame({"loss": loss}, index=dates[sl])
    rows = []
    for m in methods:
        for q in LEVELS:
            v, e = var[(m, q)][sl], es[(m, q)][sl]
            forecasts[f"VaR{q}_{m}"] = v
            forecasts[f"ES{q}_{m}"] = e
            hits = loss > v
            lr_uc, p_uc = kupiec_pof(hits, q)
            ch = christoffersen(hits, q)
            rows.append({"asset": asset, "method": m, "level": q, "n_days": hits.size,
                         "expected": (1 - q) * hits.size, "violations": int(hits.sum()),
                         "rate": hits.mean(), "kupiec_p": p_uc, "christ_ind_p": ch["p_ind"],
                         "christ_cc_p": ch["p_cc"],
                         # ES check: mean realised loss / mean ES on violation days
                         "es_ratio": loss[hits].mean() / e[hits].mean() if hits.any() else np.nan,
                         "mean_VaR": v.mean()})
    return forecasts, pd.DataFrame(rows)


def plot_backtest(asset, forecasts, summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    q = 0.99
    ax.plot(forecasts.index, forecasts.loss, color="#bdbcb6", lw=0.5, label="daily loss (%)")
    for m, color in [("HS (raw)", COLORS["reference"]), ("GPD (POT)", "#3d3d3a"),
                     ("CS-powerlaw", COLORS["powerlaw"])]:
        ax.plot(forecasts.index, forecasts[f"VaR{q}_{m}"], color=color, lw=0.8,
                label=f"99% VaR, {m}")
    ax.set_ylim(-1, np.nanpercentile(forecasts.loss, 99.97) * 1.1)
    ax.set_ylabel("loss (%)")
    ax.set_title(f"{asset}: one-day 99% VaR forecasts (out of sample)")
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    methods = summary.method.unique()
    ypos = np.arange(len(methods))
    for j, (q, mk) in enumerate([(0.99, "o"), (0.995, "s")]):
        s = summary[summary.level == q].set_index("method").loc[methods]
        ax.scatter(s.rate / (1 - q), ypos + (j - 0.5) * 0.25, marker=mk, s=28,
                   color=[COLORS["powerlaw"] if m.startswith("CS") else "#3d3d3a" for m in methods],
                   label=f"{q:.1%} VaR")
    ax.axvline(1, color="black", ls=":", lw=1)
    ax.set_yticks(ypos, methods)
    ax.invert_yaxis()
    ax.set_xlabel("violation rate / expected rate")
    ax.set_title("Coverage (1 = exact)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"backtest_{asset}.png")
    plt.close(fig)


def main():
    asset = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    forecasts, summary = run_backtest(asset)
    forecasts.to_csv(RES / f"backtest_{asset}.csv.gz")
    summary.to_csv(RES / f"backtest_{asset}_summary.csv", index=False)
    plot_backtest(asset, forecasts, summary)
    pd.set_option("display.width", 200)
    print()
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
