"""Step 2: download and cache daily prices from Yahoo Finance (free, via yfinance).

Writes data/<asset>.csv and prints basic data-quality checks.
Run:  python scripts/02_download_data.py [--refresh]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailsurr.data import ASSETS, load_prices, log_returns  # noqa: E402


def main(refresh=False):
    for name in ASSETS:
        close = load_prices(name, refresh=refresh)
        r = log_returns(close)
        big = r[r.abs() > 8 * r.std()]
        print(f"{name:7s} {close.index[0].date()} -> {close.index[-1].date()}  "
              f"n_returns={r.size}  std={r.std():.3f}%  min={r.min():.2f}%  "
              f"max={r.max():.2f}%  zero-returns={int((r == 0).sum())}")
        for day, val in big.items():
            print(f"         |r| > 8 sd: {day.date()} {val:+.2f}%")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
