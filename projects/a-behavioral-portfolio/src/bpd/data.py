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


def moments(rets: pd.DataFrame, shrink: float = None):
    """Monthly mean vector and covariance matrix, the mu and Sigma of deck 002.

    The covariance is the sample covariance, because Chopra and Ziemba (1993), on page 20
    of that deck, rank equal relative errors in means, variances and covariances at
    10 : 2 : 1 in certainty-equivalent loss, so cleaning the covariance is work on the
    least damaging input.

    The mean is shrunk toward the cross-sectional grand mean, deliberately and with the
    intensity reported, because page 27 of the same deck shows that a binding long-only
    constraint already shrinks the inputs silently and prescribes doing it on purpose
    instead.  At shrink = 0 the estimate is the raw sample mean and at shrink = 1 every
    asset is given the same mean, which removes the cross-sectional signal entirely.
    """
    mu = rets.mean().to_numpy()
    sigma = rets.cov().to_numpy()
    shrink = config.MU_SHRINK if shrink is None else shrink
    return shrink_mean(mu, shrink), sigma


def shrink_mean(mu: np.ndarray, shrink: float) -> np.ndarray:
    """Pull the sample mean toward its cross-sectional average by `shrink`."""
    mu = np.asarray(mu, dtype=float)
    return (1.0 - shrink) * mu + shrink * mu.mean()


def equalise_means(rets: np.ndarray) -> np.ndarray:
    """Shift every asset's returns so they share one expected return.

    The covariance matrix and the month-to-month structure are untouched, so bootstrapping
    from this matrix produces scenarios that differ from the main ones in exactly one
    respect: no asset carries a drift advantage over any other.  It is the test of whether
    a policy's ranking came from the policy or from the decade the sample covers, and it
    is the course's own perturbation exercise of deck "002 mean variance" page 23 applied
    to the whole cross-section at once rather than to one asset.
    """
    rets = np.asarray(rets, dtype=float)
    return rets - rets.mean(axis=0) + rets.mean()


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
