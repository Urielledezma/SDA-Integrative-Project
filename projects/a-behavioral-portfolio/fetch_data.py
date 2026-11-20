"""Download the price history once and cache it, so the report reproduces offline.

    python fetch_data.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from bpd import config, data  # noqa: E402

prices = data.download_prices()
data.save_prices(prices)
rets = data.monthly_returns(prices)
print(f"daily prices   {prices.shape[0]} rows, {prices.shape[1]} tickers, "
      f"{prices.index[0].date()} to {prices.index[-1].date()}")
print(f"monthly returns {rets.shape[0]} months, {rets.index[0].date()} to {rets.index[-1].date()}")
print(f"missing values  {int(prices.isna().sum().sum())}")
print(f"saved to        {data.PRICE_FILE}")
