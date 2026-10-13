"""Download, cache and pre-process daily price data (free Yahoo Finance data)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Cached prices live in <repo>/data unless TAILSURR_DATA_DIR points elsewhere.
DATA_DIR = Path(os.environ.get("TAILSURR_DATA_DIR",
                               Path(__file__).resolve().parents[2] / "data"))

# name -> (source, ticker, first date used).
#  * ^GSPC starts in 1927 on Yahoo; we use 1950 onwards because the early index
#    history is less reliable.
#  * Yahoo's EURUSD=X series contains obvious bad ticks (e.g. +16% then -14% on
#    8-9 Dec 2008), so EUR/USD comes from FRED series DEXUSEU instead (Federal
#    Reserve H.10 noon buying rates in New York, USD per EUR).
ASSETS = {
    "SP500": ("yahoo", "^GSPC", "1950-01-01"),
    "BTC": ("yahoo", "BTC-USD", "2014-09-17"),
    "EURUSD": ("fred", "DEXUSEU", "1999-01-01"),
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
END_DATE = "2026-10-13"  # exclusive: the pilot uses closes up to 2026-10-12


def load_prices(name, refresh=False):
    """Daily adjusted close for one asset, cached as CSV under data/."""
    source, ticker, start = ASSETS[name]
    path = DATA_DIR / f"{name}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    if source == "yahoo":
        import yfinance as yf  # imported lazily: only needed when downloading

        df = yf.download(ticker, start=start, end=END_DATE, interval="1d",
                         auto_adjust=True, progress=False)
        close = df["Close"].squeeze()
    else:  # FRED: missing days are blank or "."
        df = pd.read_csv(FRED_URL.format(ticker), index_col=0, parse_dates=True,
                         na_values=["."])
        close = df[ticker]
        close = close[(close.index >= start) & (close.index < END_DATE)]
    close = close.rename("close").dropna()
    close.index.name = "date"
    close = close[close > 0]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    close.to_csv(path)
    return close


def log_returns(close):
    """Daily log returns in percent."""
    return 100 * np.log(close).diff().dropna()


def garch_standardised_residuals(returns):
    """Fit a constant-mean GARCH(1,1) by Gaussian quasi-maximum likelihood.

    Returns (z, fit) where z_t = (r_t - mu) / sigma_t are the standardised
    residuals: approximately i.i.d. if the GARCH filter captures the volatility
    clustering (the McNeil & Frey 2000 'filter then model the tail' idea).
    """
    from arch import arch_model

    fit = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1,
                     dist="normal", rescale=False).fit(disp="off")
    z = (fit.resid / fit.conditional_volatility).dropna()
    return z, fit
