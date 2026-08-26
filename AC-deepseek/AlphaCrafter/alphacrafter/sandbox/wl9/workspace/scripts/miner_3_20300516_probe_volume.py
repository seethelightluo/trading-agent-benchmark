"""Probe volume data availability for a liquidity factor idea (miner_3 cycle 2030-05-16)."""
import pandas as pd
import numpy as np
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX',
         'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

for sym in WATCH:
    try:
        df = get_stock_daily_data(symbol=sym, days=2500)
        if df is None or len(df) == 0:
            df = get_index_daily_data(symbol=sym, days=2500)
        if df is None or len(df) == 0:
            print(sym, "NO DATA")
            continue
        if 'volume' not in df.columns or 'vol' not in df.columns:
            pass
        vol = df['volume'] if 'volume' in df.columns else (df['vol'] if 'vol' in df.columns else None)
        if vol is None:
            print(f"{sym:9s} NO VOLUME COLUMN; cols={list(df.columns)}")
            continue
        vol = pd.to_numeric(vol, errors='coerce')
        nz = (vol.dropna() > 0).mean()
        med = vol.dropna().median()
        print(f"{sym:9s} rows={len(df):5d} vol_nonzero_frac={nz:.3f} median={med:.6g} "
              f"n_null={vol.isna().sum()} last5={list(vol.iloc[-5:].values)}")
    except Exception as e:
        print(sym, "ERR", e)