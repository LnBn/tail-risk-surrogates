"""Step 1: validate the constrained-surrogate samplers on synthetic i.i.d. data.

(a) Commutativity: statistics of surrogates of process samples have the same
    distribution as statistics of fresh process samples.
(b) Exactness: p-values are uniform when the null family is true.
(c) Power: rejection rates when the data come from a different family.
(d) Unbiasedness: estimating the probability of an extreme larger than
    anything typically seen in the sample.

Outputs: figures/validation_*.png and results/validation_*.csv
Run:     python scripts/01_validate_synthetic.py   (about 10-15 min on 6 cores)
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailsurr import (constrained_surrogates, fit_mle, sample_family,  # noqa: E402
                      surrogate_pvalue, typical_surrogates, top_hill)
from tailsurr.fitting import sf  # noqa: E402
from tailsurr.plotting import COLORS, LABELS, plt  # noqa: E402
from tailsurr.stats import ks_to_fixed_fit  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"
XMIN = 1.0
SEED = 20261013
# same parameters as Table III of Moore & Altmann (2026)
TRUE = {"powerlaw": {"alpha": 2.5},
        "lognormal": {"mu": -1.0, "sigma": 1.0},
        "exponential": {"lam": 1.0}}
NULLS = list(TRUE)
PROCESSES = NULLS + ["student_t"]
T_DF, T_TAIL = 3, 0.10  # Student-t(3) values above their 90% quantile
N_WORKERS = 6


def generate(process, size, rng):
    """i.i.d. samples on [XMIN, inf) from one of the test processes."""
    if process == "student_t":
        # finance-like tail: asymptotically a power law (zeta = 3) but not
        # exactly one above the threshold.  Inverse-CDF sampling, rescaled
        # so the threshold sits at XMIN.
        q = stats.t.ppf(1 - T_TAIL, T_DF)
        u = rng.uniform(1 - T_TAIL, 1.0, size)
        return stats.t.ppf(u, T_DF) / q * XMIN
    return sample_family(process, TRUE[process], XMIN, size, rng)


def true_sf(process, v):
    if process == "student_t":
        q = stats.t.ppf(1 - T_TAIL, T_DF)
        return stats.t.sf(v * q / XMIN, T_DF) / T_TAIL
    return sf(v, process, TRUE[process], XMIN)


def true_quantile(process, p_exceed):
    """v with P(X > v) = p_exceed, by root finding on the survival function."""
    from scipy.optimize import brentq
    return brentq(lambda v: true_sf(process, v) - p_exceed, XMIN, 1e12)


# ---------------------------------------------------------------------------
# (a) commutativity
# ---------------------------------------------------------------------------

def commutativity(rng, n=50, reps=4000):
    rows, curves = [], {}
    for fam in NULLS:
        x = generate(fam, (reps, n), rng)
        fresh = generate(fam, (reps, n), rng)
        cons = constrained_surrogates(x, fam, XMIN, 1, rng=rng)[:, 0]
        typ = typical_surrogates(x, fam, XMIN, 1, rng=rng)[:, 0]
        for stat_name, f in [("log10 max", lambda a: np.log10(a.max(-1))),
                             ("top-10% Hill", lambda a: top_hill(a, 0.1))]:
            ref = f(fresh)
            for method, s in [("constrained", f(cons)), ("typical", f(typ))]:
                p = stats.ks_2samp(ref, s).pvalue
                rows.append({"family": fam, "statistic": stat_name, "method": method,
                             "mean_fresh": ref.mean(), "mean_surrogate": s.mean(),
                             "ks2_pvalue": p})
                curves[(fam, stat_name, method)] = (np.sort(ref), np.sort(s), p)
    return pd.DataFrame(rows), curves


def plot_commutativity(curves, n):
    stat_names = ["log10 max", "top-10% Hill"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    probs = np.linspace(0.005, 0.995, 199)
    for c, fam in enumerate(NULLS):
        for r, st in enumerate(stat_names):
            ax = axes[r, c]
            for method, ls in [("constrained", "-"), ("typical", "--")]:
                ref, s, p = curves[(fam, st, method)]
                ax.plot(np.quantile(ref, probs), np.quantile(s, probs), ls,
                        color=COLORS[fam] if method == "constrained" else COLORS["reference"],
                        label=f"{method} (KS p={p:.2f})")
            lo, hi = ax.get_xlim()
            ax.plot([lo, hi], [lo, hi], ":", color="black", lw=1)
            ax.set_title(f"{LABELS[fam]}: {st}")
            ax.set_xlabel("quantiles, fresh samples")
            if c == 0:
                ax.set_ylabel("quantiles, surrogates")
            ax.legend(loc="upper left")
    fig.suptitle(f"(a) Commutativity: surrogates of samples vs fresh samples (N={n})")
    fig.tight_layout()
    fig.savefig(FIG / "validation_commutativity.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (b, c) size and power
# ---------------------------------------------------------------------------

def _one_cell(args):
    """All tests for one (process, null family, N) cell of the grid."""
    process, null, n, reps, n_surr, seed = args
    rng = np.random.default_rng(seed)
    x = generate(process, (reps, n), rng)
    surr = constrained_surrogates(x, null, XMIN, n_surr, rng=rng)  # (reps, M, n)
    out = []

    def add(method, stat, p):
        for rep, pv in enumerate(np.atleast_1d(p)):
            out.append((process, null, n, method, stat, rep, float(pv)))

    add("constrained", "max", surrogate_pvalue(x.max(-1), surr.max(-1)))
    add("constrained", "top_hill", surrogate_pvalue(top_hill(x), top_hill(surr)))
    # KS to the MLE fit: the fit is shared by a dataset and its surrogates
    ks_p = []
    for d in range(reps):
        params = fit_mle(x[d], null, XMIN)
        ks_p.append(surrogate_pvalue(ks_to_fixed_fit(x[d], null, XMIN, params),
                                     ks_to_fixed_fit(surr[d], null, XMIN, params),
                                     "greater"))
    add("constrained", "ks", np.array(ks_p))
    if process == null:  # typical (parametric bootstrap) surrogates, for size only
        typ = typical_surrogates(x, null, XMIN, n_surr, rng=rng)
        add("typical", "max", surrogate_pvalue(x.max(-1), typ.max(-1)))
        add("typical", "top_hill", surrogate_pvalue(top_hill(x), top_hill(typ)))
    return out


def size_and_power(ns=(50, 100, 250, 500), reps=400, n_surr=99):
    cells = list(product(PROCESSES, NULLS, ns))
    seeds = np.random.SeedSequence(SEED + 1).generate_state(len(cells))
    tasks = [(p, nu, n, reps, n_surr, int(s)) for (p, nu, n), s in zip(cells, seeds)]
    rows = []
    with ProcessPoolExecutor(N_WORKERS) as pool:
        for res in pool.map(_one_cell, tasks):
            rows.extend(res)
    return pd.DataFrame(rows, columns=["process", "null", "N", "method", "statistic",
                                       "rep", "pvalue"])


STAT_STYLE = {"max": ("#6250d6", "o", "max (two-sided)"),
              "top_hill": ("#e87ba4", "s", "top-10% Hill (two-sided)"),
              "ks": ("#3d3d3a", "^", "KS to MLE fit (upper)")}


def plot_pvalue_uniformity(pv, n=100):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    grid = np.linspace(0, 1, 101)
    for ax, fam in zip(axes, NULLS):
        sub = pv[(pv.process == fam) & (pv.null == fam) & (pv.N == n)]
        for (method, stat), g in sub.groupby(["method", "statistic"]):
            ecdf = (g.pvalue.values[:, None] <= grid).mean(0)
            color, marker, label = STAT_STYLE[stat]
            ls = "-" if method == "constrained" else "--"
            ax.step(grid, ecdf, ls, where="post", color=color,
                    label=f"{label}, {method}")
        ax.plot([0, 1], [0, 1], ":", color="black", lw=1)
        ax.set_title(f"data and null: {LABELS[fam]}")
        ax.set_xlabel("nominal level")
    axes[0].set_ylabel("fraction of p-values below level")
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.suptitle(f"(b) p-values under the null (N={n}, 400 datasets, 99 surrogates)")
    fig.tight_layout()
    fig.savefig(FIG / "validation_pvalues.png")
    plt.close(fig)


def plot_power(pv, level=0.05):
    rej = (pv.assign(reject=pv.pvalue <= level)
             .groupby(["process", "null", "N", "method", "statistic"]).reject.mean()
             .reset_index())
    fig, axes = plt.subplots(len(PROCESSES), len(NULLS), figsize=(9, 9.5),
                             sharex=True, sharey=True)
    for r, proc in enumerate(PROCESSES):
        for c, null in enumerate(NULLS):
            ax = axes[r, c]
            sub = rej[(rej.process == proc) & (rej.null == null)]
            for (method, stat), g in sub.groupby(["method", "statistic"]):
                color, marker, label = STAT_STYLE[stat]
                ax.plot(g.N, g.reject, "-" if method == "constrained" else "--",
                        color=color, marker=marker, ms=4,
                        label=f"{label}" + ("" if method == "constrained" else ", typical"))
            if proc == null:  # 95% binomial band around the nominal size
                reps = pv[(pv.process == proc) & (pv.null == null)].rep.nunique()
                half = 1.96 * np.sqrt(level * (1 - level) / reps)
                ax.axhspan(level - half, level + half, color=COLORS["reference"], alpha=0.25, lw=0)
                ax.set_facecolor("#fbf4f1")
            ax.axhline(level, color=COLORS["reference"], lw=1, ls=":")
            ax.set_xscale("log")
            ax.set_xticks([50, 100, 250, 500], ["50", "100", "250", "500"])
            ax.minorticks_off()
            if r == 0:
                ax.set_title(f"null: {LABELS[null]}", color=COLORS[null])
            if c == 0:
                ax.set_ylabel(f"data: {LABELS[proc]}\nrejection rate")
            if r == len(PROCESSES) - 1:
                ax.set_xlabel("tail sample size N")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"(c) Rejection rate at the {level:.0%} level (shaded panels: null true, "
                 "rate = size)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "validation_power.png")
    plt.close(fig)
    return rej


# ---------------------------------------------------------------------------
# (d) unbiased estimate of an extreme-event probability
# ---------------------------------------------------------------------------

def extreme_probability(rng, n=50, reps=2000, n_surr=200):
    """Estimate p* = P(X > v*) with v* the true 1 - 1/(10N) quantile.

    v* exceeds the sample maximum in ~90% of samples, so the empirical
    frequency is nearly always 0.  Compare the constrained-surrogate estimate
    (fraction of surrogate values above v*) with the MLE plug-in.
    """
    rows = []
    p_star = 1.0 / (10 * n)
    for fam in NULLS:
        v = true_quantile(fam, p_star)
        x = generate(fam, (reps, n), rng)
        surr = constrained_surrogates(x, fam, XMIN, n_surr, rng=rng)
        est_surr = (surr > v).mean(axis=(1, 2))
        est_mle = np.array([sf(v, fam, fit_mle(xi, fam, XMIN), XMIN) for xi in x])
        est_emp = (x > v).mean(axis=1)
        for name, est in [("constrained surrogates", est_surr),
                          ("MLE plug-in", est_mle), ("empirical frequency", est_emp)]:
            rows.append({"family": fam, "estimator": name, "true_p": p_star,
                         "mean_estimate": est.mean(),
                         "se_mean": est.std(ddof=1) / np.sqrt(reps),
                         "relative_bias": est.mean() / p_star - 1,
                         "relative_rmse": np.sqrt(np.mean((est - p_star) ** 2)) / p_star,
                         "frac_zero": np.mean(est == 0)})
    return pd.DataFrame(rows)


def plot_extreme_probability(df):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ests = ["constrained surrogates", "MLE plug-in", "empirical frequency"]
    for c, fam in enumerate(NULLS):
        for k, est in enumerate(ests):
            row = df[(df.family == fam) & (df.estimator == est)].iloc[0]
            ratio = row.mean_estimate / row.true_p
            err = 1.96 * row.se_mean / row.true_p
            ax.errorbar(c + (k - 1) * 0.22, ratio, yerr=err, fmt="os^"[k], ms=6,
                        color=COLORS[fam], mfc=COLORS[fam] if k == 0 else "white", capsize=3)
    ax.axhline(1, color="black", lw=1, ls=":")
    ax.set_xticks(range(3), [LABELS[f] for f in NULLS])
    ax.set_ylabel("mean estimate / true probability")
    ax.set_title("(d) Estimating P(X > v*), v* = true 99.8% quantile, N=50 "
                 "(markers: o surrogate, s MLE, ^ empirical)")
    fig.tight_layout()
    fig.savefig(FIG / "validation_extreme_probability.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("(a) commutativity ...", flush=True)
    comm, curves = commutativity(rng)
    comm.to_csv(RES / "validation_commutativity.csv", index=False)
    plot_commutativity(curves, 50)
    print(comm.round(4).to_string(index=False))

    print("(d) extreme-event probability ...", flush=True)
    ext = extreme_probability(rng)
    ext.to_csv(RES / "validation_extreme_probability.csv", index=False)
    plot_extreme_probability(ext)
    print(ext.round(4).to_string(index=False))

    print("(b, c) size and power grid ...", flush=True)
    pv = size_and_power()
    pv.to_csv(RES / "validation_pvalues.csv.gz", index=False)
    plot_pvalue_uniformity(pv)
    rej = plot_power(pv)
    rej.to_csv(RES / "validation_rejection_rates.csv", index=False)
    print(rej.pivot_table(index=["process", "null", "statistic", "method"],
                          columns="N", values="reject").round(3).to_string())

    # uniformity of null p-values (KS against the discrete uniform grid)
    null = pv[pv.process == pv.null]
    unif = (null.groupby(["null", "N", "method", "statistic"]).pvalue
                .apply(lambda p: stats.kstest(p, "uniform").pvalue).rename("ks_uniform_p"))
    unif.reset_index().to_csv(RES / "validation_uniformity.csv", index=False)


def replot():
    """Redraw (b) and (c) from saved p-values without re-running the simulations."""
    pv = pd.read_csv(RES / "validation_pvalues.csv.gz")
    plot_pvalue_uniformity(pv)
    plot_power(pv)


if __name__ == "__main__":
    replot() if "--replot" in sys.argv else main()
