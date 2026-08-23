#!/usr/bin/env python3
"""Check data availability across watchlist instruments."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd
import json

account = get_account_dict()
watchlist = account.get('watch_list', [])
print(f"Current date inferred from data shape")
print(f"Net assets: {account.get('net_assets', 'N/A')}")
print(f"Watchlist: {watchlist}")
print()

# Check data for each watchlist symbol
for sym in watchlist:
    try:
        df = get_index_daily_data(sym, 800)
        if df is None:
            df = get_stock_daily_data(sym, 800)
        if df is not None:
            last_date = df['date'].iloc[-1]
            n_days = len(df)
            close = df['close'].iloc[-1]
            vol = df['volume'].iloc[-1] if 'volume' in df.columns else 'N/A'
            print(f"{sym:12s}: {n_days:4d} days, last={last_date}, close={close:.2f}, vol={vol}")
        else:
            print(f"{sym:12s}: NO DATA")
    except Exception as e:
        print(f"{sym:12s}: ERROR: {e}")

# Also check VIX, DXY, USDCNY as observation signals
for sym in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    try:
        df = get_index_daily_data(sym, 800)
        if df is not None:
            last_date = df['date'].iloc[-1]
            close = df['close'].iloc[-1]
            print(f"[OBS] {sym:12s}: last={last_date}, close={close:.2f}")
        else:
            print(f"[OBS] {sym:12s}: NO DATA")
    except Exception as e:
        print(f"[OBS] {sym:12s}: ERROR: {e}")