import pandas as pd, numpy as np, json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

watch = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
for s in watch:
    df = get_stock_daily_data(symbol=s, days=600)
    v = None
    if df is not None:
        v = df['volume'].notna().sum() if 'volume' in df.columns else 0
        hasv = len(df) if df is not None else 0
        print(s, 'len', len(df), 'vol_nonnull', v, 'last_date', df['date'].iloc[-1] if df is not None else None)
    else:
        df = get_index_daily_data(symbol=s, days=600)
        v = df['volume'].notna().sum() if (df is not None and 'volume' in df.columns) else 0
        print(s, 'len', len(df), 'vol_nonnull', v, 'last_date', df['date'].iloc[-1] if df is not None else None)