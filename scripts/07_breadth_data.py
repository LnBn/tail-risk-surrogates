"""Breadth step 1: download the 22-asset universe and run the data-quality screen.

Writes data/universe/<asset>.csv (git-ignored, raw levels), and
results/breadth_data_quality.csv (one row per asset) and
results/breadth_data_events.csv (every flagged event and what was done).
Run:  python scripts/07_breadth_data.py [--refresh]
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailsurr.quality import screen  # noqa: E402
from tailsurr.universe import UNIVERSE, load_level, to_returns  # noqa: E402

RES = ROOT / "results"


def main(refresh=False):
    rows, events = [], []
    for a in UNIVERSE:
        level = load_level(a, refresh=refresh)
        r_raw = to_returns(level, a.kind)
        r, summary, ev = screen(r_raw, a.kind, a.name)
        rows.append({"asset": a.name, "class": a.asset_class, "source": a.source,
                     "ticker": a.ticker, "kind": a.kind, **summary,
                     "years": round((r.index[-1] - r.index[0]).days / 365.25, 1),
                     "note": a.note})
        if len(ev):
            events.append(ev.assign(asset=a.name))
    q = pd.DataFrame(rows)
    q.to_csv(RES / "breadth_data_quality.csv", index=False)
    ev = pd.concat(events)[["asset", "date", "check", "detail"]] if events else pd.DataFrame()
    ev.to_csv(RES / "breadth_data_events.csv", index=False)
    pd.set_option("display.width", 250)
    print(q.drop(columns="note").to_string(index=False))
    print()
    print(ev.groupby(["asset", "check"]).size().to_string())


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
