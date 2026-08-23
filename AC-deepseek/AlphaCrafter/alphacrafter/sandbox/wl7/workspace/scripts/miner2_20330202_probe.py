import sys, glob, os
sys.path.insert(0, "scripts")
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

def probe(sym, days=4000):
    df = None
    try:
        df = get_stock_daily_data(symbol=sym, days=days)
    except Exception as e:
        pass
    if df is None:
        try:
            df = get_index_daily_data(symbol=sym, days=days)
        except Exception:
            return None
    return df

watch = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
print("=== WATCHLIST SPAN ===")
min_date = None
max_date = None
for s in watch:
    df = probe(s)
    if df is None:
        print(s, "NO DATA")
        continue
    d0 = str(df['date'].iloc[0])[:10]
    d1 = str(df['date'].iloc[-1])[:10]
    print(f"{s}: rows={len(df)} {d0}..{d1}")
    if max_date is None or len(df) > len(maxdf):
        maxdf = df
print("max date:", str(maxdf['date'].iloc[-1])[:10])
print("rows for max:", len(maxdf))

obs = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
print("=== OBSERVATION SPAN (get_index) ===")
for s in obs:
    df = probe(s)
    if df is None:
        print(s, "NO DATA via get_index")
        # try index_data
        p = f"../persistent/index_data/{s}.csv"
        if os.path.exists(p):
            print("  csv exists", p)
        continue
    d0 = str(df['date'].iloc[0])[:10]
    d1 = str(df['date'].iloc[-1])[:10]
    print(f"{s}: rows={len(df)} {d0}..{d1}")