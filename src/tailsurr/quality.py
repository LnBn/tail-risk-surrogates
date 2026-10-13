"""Data-quality screen for daily return series.

Three problems matter for tail analysis:

1. Bad ticks: a single wrong print shows up as a huge move immediately
   reversed (spike-and-reverse).  Flag day t when |r_t| and |r_{t+1}| both
   exceed SPIKE_K local standard deviations, have opposite signs, and nearly
   cancel.  Real crash-and-rebound days can also trigger this, so every flag
   was inspected by hand: those listed in GENUINE are kept; all others are
   cleaned by dropping the bad level, i.e. merging r_t and r_{t+1} into one
   return dated t+1.
2. Stale prices: runs of identical closes (zero returns) of length >= STALE_RUN.
   Cleaning: drop those zero returns (prices only; small yield changes are
   genuinely zero at 1bp resolution).
3. Gaps: consecutive observations more than GAP_DAYS calendar days apart.
   Cleaning: drop the return spanning the gap (it is a multi-week return).

Ties (many identical values among the largest moves) are reported because the
surrogate methods assume continuous data.
"""

import numpy as np
import pandas as pd

SPIKE_K = 6.0      # local standard deviations
CANCEL = 0.25      # |r_t + r_{t+1}| < CANCEL * max(|r_t|, |r_{t+1}|)

# flagged spike-and-reverse pairs checked by hand and confirmed as real moves
GENUINE = {
    ("SP500", "2020-03-12"): "COVID crash: -9.5% on 12 Mar, +9.3% on 13 Mar 2020",
    ("JPM", "2020-03-13"): "COVID crash: +18% on 13 Mar, -15% on 16 Mar 2020",
    ("NASDAQ", "1987-10-20"): "October 1987 crash and rebound",
    ("HANGSENG", "1989-05-22"): "Tiananmen-period crash and rebound, May 1989",
    ("HANGSENG", "1997-10-28"): "Asian financial crisis: -13.7% on 28 Oct, +18.8% on 29 Oct 1997",
    ("XOM", "1962-05-28"): "28 May 1962 'flash crash' and next-day recovery",
}
STALE_RUN = 5
GAP_DAYS = 10


def local_scale(r, window=250):
    """Rolling sqrt(pi/2) * mean |r|: a local standard deviation.

    (A median-based scale is more robust to spikes but collapses to zero for
    series that are mostly unchanged, e.g. 1960s yields at 1bp resolution.)
    """
    s = np.sqrt(np.pi / 2) * r.abs().rolling(window, min_periods=50, center=True).mean()
    return s.bfill().ffill()


def find_spikes(r):
    """Dates t of spike-and-reverse pairs (t, t+1)."""
    s = local_scale(r).values
    x = r.values
    big = np.abs(x) > SPIKE_K * s
    flags = []
    for t in range(len(x) - 1):
        if big[t] and np.abs(x[t + 1]) > SPIKE_K * s[t] and x[t] * x[t + 1] < 0:
            if abs(x[t] + x[t + 1]) < CANCEL * max(abs(x[t]), abs(x[t + 1])):
                flags.append(t)
    return [r.index[t] for t in flags]


def stale_runs(r):
    """List of (start_date, end_date, length) runs of >= STALE_RUN zero returns."""
    zero = (r.values == 0)
    runs, t = [], 0
    while t < len(zero):
        if zero[t]:
            u = t
            while u + 1 < len(zero) and zero[u + 1]:
                u += 1
            if u - t + 1 >= STALE_RUN:
                runs.append((r.index[t], r.index[u], u - t + 1))
            t = u + 1
        else:
            t += 1
    return runs


def screen(r, kind="log", asset=""):
    """Screen and clean a return series.

    Returns (clean_returns, summary dict, events DataFrame).
    """
    events = []
    r = r.copy()

    # 1. spike-and-reverse: merge the two returns around the bad level
    for day in find_spikes(r):
        i = r.index.get_loc(day)
        a, b = r.iloc[i], r.iloc[i + 1]
        reason = GENUINE.get((asset, str(day.date())))
        if reason:
            events.append({"date": day.date(), "check": "spike-and-reverse (kept)",
                           "detail": f"{a:+.2f} then {b:+.2f}; verified genuine: {reason}"})
            continue
        events.append({"date": day.date(), "check": "spike-and-reverse",
                       "detail": f"{a:+.2f} then {b:+.2f}; merged into {a + b:+.2f}"})
        r.iloc[i + 1] = a + b
        r = r.drop(day)

    # 2. stale prices
    n_stale = 0
    if kind == "log":
        for start, end, length in stale_runs(r):
            events.append({"date": start.date(), "check": "stale run",
                           "detail": f"{length} zero returns to {end.date()}; dropped"})
            n_stale += length
            r = r.drop(r.loc[start:end].index)

    # 3. calendar gaps
    gaps = r.index.to_series().diff().dt.days
    gap_days = gaps[gaps > GAP_DAYS].index
    for day in gap_days:
        events.append({"date": day.date(), "check": "gap",
                       "detail": f"{int(gaps[day])} calendar days since previous "
                                 f"observation ({r[day]:+.2f}); dropped"})
    r = r.drop(gap_days)

    # ties among the largest 5% of absolute moves
    top = r.abs().sort_values().iloc[-max(1, len(r) // 20):]
    tie_frac = 1 - top.nunique() / len(top)

    summary = {"n_returns": len(r), "first": r.index[0].date(), "last": r.index[-1].date(),
               "spikes_merged": sum(e["check"] == "spike-and-reverse" for e in events),
               "spikes_kept": sum(e["check"] == "spike-and-reverse (kept)" for e in events),
               "stale_returns_dropped": n_stale, "gaps_dropped": len(gap_days),
               "zero_return_frac": float((r == 0).mean()),
               "tie_frac_top5pct": float(tie_frac)}
    return r, summary, pd.DataFrame(events, columns=["date", "check", "detail"])
