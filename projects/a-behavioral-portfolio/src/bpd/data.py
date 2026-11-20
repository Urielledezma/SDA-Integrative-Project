"""Prices, monthly returns, and the block bootstrap that generates the scenarios.

This is step 4 of the course's six-step process, the uncertainty model.  Deck "005 SDA"
page 16 lists three ways to model uncertainty, and the brief prescribes the second one,
resampling from observed history.
"""
import pathlib

import numpy as np
import pandas as pd

from . import config

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"
PRICE_FILE = DATA_DIR / "prices.csv"


def download_prices(tickers=None, start=None, end=None) -> pd.DataFrame:
    import yfinance as yf

    tickers = tickers or config.TICKERS
    frame = yf.download(tickers, start=start or config.DATA_START,
                        end=end or config.DATA_END, interval="1d",
                        auto_adjust=True, progress=False)
    return frame["Close"][tickers]


def save_prices(prices: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(PRICE_FILE, float_format="%.6f")


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    return prices[config.TICKERS]


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns from month-end prices.

    Simple rather than log returns, because the wealth transition multiplies wealth by
    1 + w'r and that identity holds only for simple returns, while a log return would
    have to be converted before it could be weighted.
    """
    return prices.resample("ME").last().pct_change().dropna()


def moments(rets: pd.DataFrame):
    """Monthly mean vector and covariance matrix, the mu and Sigma of deck 002."""
    return rets.mean().to_numpy(), rets.cov().to_numpy()


def block_bootstrap(rets: np.ndarray, n_paths: int, horizon: int, seed: int,
                    block_min: int = None, block_max: int = None) -> np.ndarray:
    """Draw n_paths scenarios of length horizon by resampling blocks of months.

    Whole rows are resampled, so the cross-section of returns within a month and the
    serial dependence inside a block both survive, which a draw of independent single
    months would destroy.  Returns an array of shape (n_paths, horizon, n_assets).
    """
    block_min = block_min or config.BLOCK_MIN
    block_max = block_max or config.BLOCK_MAX
    n_obs, n_assets = rets.shape
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, horizon, n_assets))
    for p in range(n_paths):
        filled = 0
        while filled < horizon:
            length = rng.integers(block_min, block_max + 1)
            start = rng.integers(0, n_obs - length + 1)
            take = min(length, horizon - filled)
            paths[p, filled:filled + take] = rets[start:start + take]
            filled += take
    return paths
