"""The 'breadth' asset universe: definitions, download/caching, return series.

Every series records its source.  Raw prices are cached under data/universe/
(or $TAILSURR_DATA_DIR/universe; git-ignored) and are never redistributed.

Return conventions
  * prices: daily log returns in percent, r_t = 100 log(P_t / P_{t-1})
  * the 10-year Treasury yield: daily *changes* in basis points (yields can be
    near zero, so log returns are meaningless)
Left tail = falls (losses for a long position; yield falls for DGS10),
right tail = rises.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# <repo>/data/universe unless TAILSURR_DATA_DIR points elsewhere.
DATA_DIR = Path(os.environ.get("TAILSURR_DATA_DIR",
                               Path(__file__).resolve().parents[2] / "data")) / "universe"
END_DATE = "2026-10-13"  # exclusive; same cut-off as the pilot
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


@dataclass(frozen=True)
class Asset:
    name: str
    asset_class: str
    source: str        # "yahoo" or "fred"
    ticker: str
    start: str         # first date used
    kind: str = "log"  # "log" (log returns, %) or "diff" (changes, bp)
    note: str = ""


UNIVERSE = [
    # equity indices (Yahoo; price indices, no dividends)
    Asset("SP500", "equity index", "yahoo", "^GSPC", "1950-01-01",
          note="pre-1950 Yahoo history dropped (less reliable)"),
    Asset("NASDAQ", "equity index", "yahoo", "^IXIC", "1971-02-05"),
    Asset("FTSE100", "equity index", "yahoo", "^FTSE", "1984-01-03"),
    Asset("DAX", "equity index", "yahoo", "^GDAXI", "1987-12-30"),
    Asset("NIKKEI225", "equity index", "yahoo", "^N225", "1965-01-05"),
    Asset("HANGSENG", "equity index", "yahoo", "^HSI", "1986-12-31"),
    Asset("EUROSTOXX50", "equity index", "yahoo", "^STOXX50E", "2007-03-30",
          note="Yahoo history starts 2007"),
    # single stocks, one per sector (Yahoo, split/dividend adjusted)
    Asset("IBM", "single stock", "yahoo", "IBM", "1962-01-02", note="technology"),
    Asset("KO", "single stock", "yahoo", "KO", "1962-01-02", note="consumer staples"),
    Asset("XOM", "single stock", "yahoo", "XOM", "1962-01-02", note="energy"),
    Asset("JPM", "single stock", "yahoo", "JPM", "1980-03-17", note="financials"),
    Asset("JNJ", "single stock", "yahoo", "JNJ", "1962-01-02", note="health care"),
    # FX (FRED H.10 noon buying rates; Yahoo FX has bad ticks).  Start in 1974,
    # after the end of Bretton Woods (1973 has stale stretches); AUD floated
    # only on 12 Dec 1983 (before that a managed rate with weeks of no change).
    Asset("EURUSD", "fx", "fred", "DEXUSEU", "1999-01-04", note="USD per EUR"),
    Asset("GBPUSD", "fx", "fred", "DEXUSUK", "1974-01-02", note="USD per GBP"),
    Asset("USDJPY", "fx", "fred", "DEXJPUS", "1974-01-02", note="JPY per USD"),
    Asset("USDCHF", "fx", "fred", "DEXSZUS", "1974-01-02", note="CHF per USD"),
    Asset("AUDUSD", "fx", "fred", "DEXUSAL", "1983-12-12",
          note="USD per AUD; starts at the float, 12 Dec 1983"),
    # commodities
    Asset("GOLD", "commodity", "yahoo", "GLD", "2004-11-18",
          note="SPDR Gold Shares ETF; gold is no longer on FRED, and GC=F "
               "front-month futures carry roll jumps"),
    Asset("BRENT", "commodity", "fred", "DCOILBRENTEU", "1987-05-20",
          note="Brent spot; WTI excluded because it settled at -37.63 USD on "
               "2020-04-20, where log returns are undefined"),
    # rates
    Asset("UST10Y", "rates", "fred", "DGS10", "1962-01-02", kind="diff",
          note="10-year constant-maturity Treasury yield; daily changes in bp"),
    # crypto (Yahoo; 7-day week)
    Asset("BTC", "crypto", "yahoo", "BTC-USD", "2014-09-17"),
    Asset("ETH", "crypto", "yahoo", "ETH-USD", "2017-11-09"),
]
BY_NAME = {a.name: a for a in UNIVERSE}


def load_level(asset, refresh=False):
    """Daily closing level (price, rate or yield) for an asset, cached as CSV."""
    path = DATA_DIR / f"{asset.name}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)["level"]
    if asset.source == "yahoo":
        import yfinance as yf

        df = yf.download(asset.ticker, start=asset.start, end=END_DATE,
                         interval="1d", auto_adjust=True, progress=False)
        level = df["Close"].squeeze()
    elif asset.source == "fred":
        df = pd.read_csv(FRED_URL.format(asset.ticker), index_col=0,
                         parse_dates=True, na_values=["."])
        level = df[asset.ticker]
    else:
        raise ValueError(asset.source)
    level = level[(level.index >= asset.start) & (level.index < END_DATE)]
    level = level.rename("level").dropna()
    level.index.name = "date"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    level.to_csv(path)
    return level


def to_returns(level, kind):
    """Log returns in % (prices) or first differences in bp (yields)."""
    if kind == "log":
        if (level <= 0).any():
            raise ValueError("non-positive price: log returns undefined")
        return (100 * np.log(level).diff()).dropna()
    if kind == "diff":
        return (100 * level.diff()).dropna()  # percent -> basis points
    raise ValueError(kind)
