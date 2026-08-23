"""miner3_20280127_probe: verify data availability through current date."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

acc = get_account_dict()
wl = acc.get('watch_list', WATCHLIST)
print("watch_list:", wl)

for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=2600)
    if df is None:
        print(f"{sym}: NONE")
        continue
    df['date'] = pd.to_datetime(df['date'])
    print(f"{sym}: n={len(df)}, {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}, last_close={df['close'].iloc[-1]:.2f}")

# VIX observation-only
vix = get_index_daily_data(symbol='VIX', days=2600)
if vix is not None:
    vix['date'] = pd.to_datetime(vix['date'])
    print(f"VIX: n={len(vix)}, {vix['date'].iloc[0].date()} -> {vix['date'].iloc[-1].date()}, last={vix['close'].iloc[-1]:.2f}")