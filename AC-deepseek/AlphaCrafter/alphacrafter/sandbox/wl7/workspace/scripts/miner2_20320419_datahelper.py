"""Shared data loader for miner_2. Loads 15 tradable instrument close series."""
import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def load_close(days=2600):
    frames = {}
    for s in WATCH:
        df = None
        try:
            df = get_index_daily_data(symbol=s, days=days)
        except Exception:
            df = None
        if df is None or len(df) == 0:
            try:
                df = get_stock_daily_data(symbol=s, days=days)
            except Exception:
                df = None
        if df is not None and len(df) > 0:
            d = df.copy()
            d['date'] = pd.to_datetime(d['date'])
            frames[s] = d.set_index('date')['close']
    px = pd.DataFrame(frames)
    px = px.sort_index()
    return px

def load_ret(px):
    return px.pct_change(fill_method=None)

if __name__ == "__main__":
    px = load_close()
    print("shape", px.shape)
    print("dates range", px.index.min(), px.index.max())
    print("coverage:", px.notna().mean())