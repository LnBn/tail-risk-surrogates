"""Breadth step 3: multiple-testing control, verdicts, pooled tests, figures.

Reads results/breadth_*.csv from step 08 and writes
  results/breadth_verdicts.csv          asset x side x family verdicts (BH, filtered)
  results/breadth_sensitivity_summary.csv  per asset x side x family: thresholds rejected
  results/breadth_pooled.csv            cross-asset Cauchy (and Fisher) combinations
  results/breadth_tail_index_meta.csv   random-effects pooled tail index
  results/breadth_summary_table.{csv,md} one row per asset x tail: everything at a glance
  figures/breadth_verdicts.png, breadth_tail_index.png, breadth_sensitivity.png
Run: python scripts/09_breadth_summary.py   (seconds)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailsurr import FAMILIES  # noqa: E402
from tailsurr.multitest import (bh_adjust, cauchy_combination, fisher_combination,  # noqa: E402
                                holm_adjust)
from tailsurr.plotting import COLORS, LABELS, plt  # noqa: E402
from tailsurr.universe import UNIVERSE  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
Q = 0.05
ASSETS = [a.name for a in UNIVERSE]
CLASS = {a.name: a.asset_class for a in UNIVERSE}
INK, LIGHT = "#3d3d3a", "#e4e3df"
SIDE_COLOR = {"left": COLORS["powerlaw"], "right": "#6250d6"}


def add_adjustments(df):
    """BH q-values and Holm-adjusted p-values across all rows of df."""
    df = df.copy()
    df["q_bh"] = bh_adjust(df.pvalue.values)
    df["p_holm"] = holm_adjust(df.pvalue.values)
    return df


def direction(row):
    """Human-readable direction of a deviation (observed vs surrogates)."""
    if row.statistic == "max":
        return "max larger than family implies" if row.z > 0 else "max smaller than family implies"
    if row.statistic == "top_hill":
        return "far tail lighter than fit" if row.z > 0 else "far tail heavier than fit"
    return "poor overall fit (KS)"


def verdicts(head):
    """One row per series x asset x side x family."""
    rows = []
    for (series, asset, side, fam), g in head.groupby(["series", "asset", "side", "family"]):
        best = g.loc[g.q_bh.idxmin()]
        rows.append({"series": series, "asset": asset, "class": CLASS[asset], "side": side,
                     "family": fam, "min_p": g.pvalue.min(), "min_q_bh": best.q_bh,
                     "min_p_holm": g.p_holm.min(),
                     "rejected_bh": best.q_bh <= Q, "rejected_holm": g.p_holm.min() <= Q,
                     "deciding_statistic": best.statistic if best.q_bh <= Q else "",
                     "direction": direction(best) if best.q_bh <= Q else ""})
    return pd.DataFrame(rows)


def sensitivity_summary(sens):
    sens = add_adjustments(sens)
    cell = (sens.groupby(["asset", "side", "family", "tail_frac"]).q_bh.min()
                .reset_index())
    cell["rejected"] = cell.q_bh <= Q
    rows = []
    for (asset, side, fam), g in cell.groupby(["asset", "side", "family"]):
        marks = []
        for f in [0.01, 0.02, 0.05, 0.10, 0.20]:
            h = g[np.isclose(g.tail_frac, f)]
            marks.append("." if h.empty else ("X" if h.rejected.iloc[0] else "o"))
        ok = g[~g.rejected].tail_frac
        rows.append({"asset": asset, "class": CLASS[asset], "side": side, "family": fam,
                     "pattern_1_2_5_10_20pct": " ".join(marks),
                     "largest_tail_frac_not_rejected": ok.max() if len(ok) else np.nan,
                     "n_rejected": int(g.rejected.sum()), "n_thresholds": len(g)})
    return cell, pd.DataFrame(rows)


def pooled_tests(head_f):
    """Cross-asset combination of filtered headline p-values."""
    rows = []
    for (side, fam, stat), g in head_f.groupby(["side", "family", "statistic"]):
        rows.append({"grouping": "all assets", "side": side, "family": fam, "statistic": stat,
                     "test": "two-sided/upper as in headline", "n": len(g),
                     "cauchy_p": cauchy_combination(g.pvalue), "fisher_p": fisher_combination(g.pvalue)})
    # directional tests of the power law: is the far tail systematically
    # lighter (top-Hill above surrogates) or heavier than a power law?
    pl = head_f[(head_f.family == "powerlaw") & (head_f.statistic == "top_hill")]
    for side, g in list(pl.groupby("side")) + [("both", pl)]:
        for alt, col in [("far tail lighter than power law", "p_greater"),
                         ("far tail heavier than power law", "p_less")]:
            rows.append({"grouping": "all assets", "side": side, "family": "powerlaw",
                         "statistic": "top_hill", "test": alt, "n": len(g),
                         "cauchy_p": cauchy_combination(g[col]), "fisher_p": fisher_combination(g[col]),
                         "n_positive_z": int((g.z > 0).sum())})
    # by asset class: all statistics and both sides together
    for (cls, fam), g in head_f.groupby(["class", "family"]):
        rows.append({"grouping": f"class: {cls}", "side": "both", "family": fam,
                     "statistic": "all", "test": "two-sided/upper as in headline", "n": len(g),
                     "cauchy_p": cauchy_combination(g.pvalue), "fisher_p": fisher_combination(g.pvalue)})
    return pd.DataFrame(rows)


def dersimonian_laird(est, se):
    """Random-effects meta-analysis.  Returns (pooled, se, tau, I^2)."""
    w = 1 / se ** 2
    fixed = np.sum(w * est) / w.sum()
    q = np.sum(w * (est - fixed) ** 2)
    df = est.size - 1
    tau2 = max(0.0, (q - df) / (w.sum() - np.sum(w ** 2) / w.sum()))
    w_re = 1 / (se ** 2 + tau2)
    pooled = np.sum(w_re * est) / w_re.sum()
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return pooled, np.sqrt(1 / w_re.sum()), np.sqrt(tau2), i2


def tail_index_meta(ti):
    f = ti[ti.series == "filtered"].copy()
    # bootstrap SE (includes threshold re-selection) from the 95% interval width
    f["se_boot"] = (f.zeta_boot_hi - f.zeta_boot_lo) / (2 * 1.96)
    rows = []
    groups = [("all assets", f)] + [(f"class: {c}", g) for c, g in f.groupby("class")]
    for name, g in groups:
        for side, h in g.groupby("side"):
            if len(h) < 2:
                continue
            pooled, se, tau, i2 = dersimonian_laird(h.zeta.values, h.se_boot.values)
            rows.append({"grouping": name, "side": side, "n_assets": len(h), "zeta_pooled": pooled,
                         "ci_lo": pooled - 1.96 * se, "ci_hi": pooled + 1.96 * se,
                         "tau_between_assets": tau, "I2": i2})
    return pd.DataFrame(rows)


def summary_table(ver, ti, sens_sum, clus, filters):
    """One row per asset x side for the filtered series."""
    v = ver[ver.series == "filtered"]
    f = ti[ti.series == "filtered"]
    chosen = filters[filters.chosen].set_index("asset").spec
    rows = []
    for a in ASSETS:
        for side in ["left", "right"]:
            t = f[(f.asset == a) & (f.side == side)].iloc[0]
            c = clus[(clus.asset == a) & (clus.side == side) & (clus.series == chosen[a])].iloc[0]
            row = {"asset": a, "class": CLASS[a], "side": side, "filter": chosen[a],
                   "k": int(t.k), "tail_pct": round(100 * t.tail_frac, 1),
                   "zeta": round(t.zeta, 2),
                   "zeta_95ci": f"[{t.zeta_boot_lo:.2f}, {t.zeta_boot_hi:.2f}]",
                   "residual_clustering_p": c.pvalue}
            for fam in FAMILIES:
                r = v[(v.asset == a) & (v.side == side) & (v.family == fam)].iloc[0]
                row[fam] = (f"rejected ({r.deciding_statistic}, q={r.min_q_bh:.3f})"
                            if r.rejected_bh else "ok")
            sp = sens_sum[(sens_sum.asset == a) & (sens_sum.side == side)
                          & (sens_sum.family == "powerlaw")].iloc[0]
            row["powerlaw_by_threshold_1_2_5_10_20pct"] = sp.pattern_1_2_5_10_20pct
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df):
    """Minimal markdown table (avoids an extra dependency)."""
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.values) + " |")
    return "\n".join(lines) + "\n"


def asymmetry_test(ti):
    """Is the left tail heavier (smaller zeta) than the right?  Sign test over assets."""
    f = ti[ti.series == "filtered"].pivot(index="asset", columns="side", values="zeta")
    d = f.left - f.right
    n_neg = int((d < 0).sum())
    return {"n_assets": len(d), "n_left_heavier": n_neg, "median_diff": float(d.median()),
            "sign_test_p": float(stats.binomtest(n_neg, len(d)).pvalue)}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def plot_verdicts(ver):
    v = ver[ver.series == "filtered"]
    cols = [(s, f) for s in ["left", "right"] for f in FAMILIES]
    mat = np.full((len(ASSETS), len(cols)), np.nan)
    qs = np.full_like(mat, np.nan)
    for i, a in enumerate(ASSETS):
        for j, (s, f) in enumerate(cols):
            r = v[(v.asset == a) & (v.side == s) & (v.family == f)]
            if len(r):
                mat[i, j] = r.rejected_bh.iloc[0]
                qs[i, j] = r.min_q_bh.iloc[0]
    fig, ax = plt.subplots(figsize=(6.6, 8.2))
    from matplotlib.colors import ListedColormap
    ax.imshow(mat, cmap=ListedColormap([LIGHT, INK]), vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = "rejected" if mat[i, j] == 1 else "ok"
            ax.text(j, i, f"{txt}\nq={qs[i, j]:.2g}", ha="center", va="center", fontsize=6,
                    color="white" if mat[i, j] == 1 else INK)
    ax.set_xticks(range(len(cols)), [f"{s} tail\n{LABELS[f]}" for s, f in cols], fontsize=7)
    ax.set_yticks(range(len(ASSETS)), [f"{a} ({CLASS[a]})" for a in ASSETS], fontsize=7)
    ax.axvline(2.5, color="white", lw=3)
    for k in range(1, len(ASSETS)):
        if CLASS[ASSETS[k]] != CLASS[ASSETS[k - 1]]:
            ax.axhline(k - 0.5, color="white", lw=3)
    ax.grid(False)
    ax.set_title("Filtered returns, CSN threshold: family rejected if any of max /\n"
                 "upper-tail Hill / KS has BH q <= 0.05 (396 tests)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "breadth_verdicts.png")
    plt.close(fig)


def plot_tail_index(ti, meta):
    f = ti[ti.series == "filtered"]
    fig, ax = plt.subplots(figsize=(6.4, 7.5))
    y = np.arange(len(ASSETS))
    for dy, side, mk in [(-0.17, "left", "o"), (0.17, "right", "s")]:
        g = f[f.side == side].set_index("asset").loc[ASSETS]
        ax.errorbar(g.zeta, y + dy, xerr=[g.zeta - g.zeta_boot_lo, g.zeta_boot_hi - g.zeta],
                    fmt=mk, ms=5, color=SIDE_COLOR[side], capsize=2, lw=1.2,
                    label=f"{side} tail ({'falls' if side == 'left' else 'rises'})")
        m = meta[(meta.grouping == "all assets") & (meta.side == side)].iloc[0]
        ax.axvspan(m.ci_lo, m.ci_hi, color=SIDE_COLOR[side], alpha=0.12, lw=0)
    ax.axvline(3, color=COLORS["reference"], ls=":", lw=1)
    ax.text(3.05, len(ASSETS) - 0.3, "zeta = 3", fontsize=7, color=COLORS["reference"])
    ax.set_yticks(y, [f"{a}" for a in ASSETS], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("tail index zeta (GARCH-filtered residuals, CSN threshold)\n"
                  "bars: 95% bootstrap CI incl. threshold re-selection; shading: pooled random-effects CI")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "breadth_tail_index.png")
    plt.close(fig)


def plot_sensitivity(cell):
    from matplotlib.colors import ListedColormap
    fracs = [0.01, 0.02, 0.05, 0.10, 0.20]
    fig, axes = plt.subplots(1, 6, figsize=(10, 6.2), sharey=True)
    for j, (side, fam) in enumerate([(s, f) for s in ["left", "right"] for f in FAMILIES]):
        ax = axes[j]
        mat = np.full((len(ASSETS), len(fracs)), np.nan)
        for i, a in enumerate(ASSETS):
            for k, fr in enumerate(fracs):
                c = cell[(cell.asset == a) & (cell.side == side) & (cell.family == fam)
                         & np.isclose(cell.tail_frac, fr)]
                if len(c):
                    mat[i, k] = c.rejected.iloc[0]
        ax.imshow(np.ma.masked_invalid(mat), cmap=ListedColormap([LIGHT, INK]), vmin=0, vmax=1,
                  aspect="auto")
        ax.set_xticks(range(len(fracs)), ["1", "2", "5", "10", "20"], fontsize=7)
        ax.set_xlabel("top x% of days", fontsize=7)
        ax.set_title(f"{side}\n{LABELS[fam]}", fontsize=8, color=COLORS[fam])
        ax.grid(False)
    axes[0].set_yticks(range(len(ASSETS)), ASSETS, fontsize=7)
    fig.suptitle("Threshold sensitivity (filtered): dark = rejected at BH q <= 0.05 across all "
                 "sensitivity tests; blank = too few points", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "breadth_sensitivity.png")
    plt.close(fig)


def main():
    head = pd.read_csv(RES / "breadth_headline.csv")
    sens = pd.read_csv(RES / "breadth_sensitivity.csv")
    ti = pd.read_csv(RES / "breadth_tail_index.csv")
    clus = pd.read_csv(RES / "breadth_clustering.csv")
    filters = pd.read_csv(RES / "breadth_filters.csv")

    # BH / Holm separately for the primary (filtered) and comparison (raw) families
    head = pd.concat([add_adjustments(g) for _, g in head.groupby("series")])
    head.to_csv(RES / "breadth_headline_adjusted.csv", index=False)
    ver = verdicts(head)
    ver.to_csv(RES / "breadth_verdicts.csv", index=False)

    cell, sens_sum = sensitivity_summary(sens)
    sens_sum.to_csv(RES / "breadth_sensitivity_summary.csv", index=False)

    head_f = head[head.series == "filtered"]
    pooled = pooled_tests(head_f)
    pooled.to_csv(RES / "breadth_pooled.csv", index=False)
    meta = tail_index_meta(ti)
    meta.to_csv(RES / "breadth_tail_index_meta.csv", index=False)

    table = summary_table(ver, ti, sens_sum, clus, filters)
    table.to_csv(RES / "breadth_summary_table.csv", index=False)
    (RES / "breadth_summary_table.md").write_text(to_markdown(table))
    asym = asymmetry_test(ti)
    pd.DataFrame([asym]).to_csv(RES / "breadth_asymmetry.csv", index=False)

    plot_verdicts(ver)
    plot_tail_index(ti, meta)
    plot_sensitivity(cell)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    for series in ["filtered", "raw"]:
        v = ver[ver.series == series]
        print(f"\n[{series}] rejections (BH q<=0.05) by family and side:")
        print(v.pivot_table(index="family", columns="side", values="rejected_bh", aggfunc="sum"))
        print(f"[{series}] Holm rejections:", int(v.rejected_holm.sum()))
    print("\nper-asset verdicts (filtered):")
    print(ver[ver.series == "filtered"].pivot_table(index="asset", columns=["side", "family"],
                                                   values="min_q_bh").round(3).loc[ASSETS])
    print("\nsensitivity patterns (filtered; o = ok, X = rejected, . = not run):")
    print(sens_sum.pivot_table(index="asset", columns=["side", "family"],
                               values="pattern_1_2_5_10_20pct", aggfunc="first").loc[ASSETS])
    print("\npooled:")
    print(pooled.round(4).to_string(index=False))
    print("\nclustering p<0.05 (of 44 asset-sides):",
          clus.assign(sig=clus.pvalue < 0.05).groupby("series").sig.sum().to_dict())
    print("filters chosen:", filters[filters.chosen].spec.value_counts().to_dict(),
          " non-converged fits:", filters[~filters.converged][["asset", "spec"]].values.tolist())
    print("asymmetry:", asym)
    print("\ntail index meta:")
    print(meta.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
