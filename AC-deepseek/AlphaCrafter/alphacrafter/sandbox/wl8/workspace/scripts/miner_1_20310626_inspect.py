import pandas as pd, numpy as np
ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]
DATA="../persistent/stock_data/"; IDX="../persistent/index_data/"
cur = pd.Timestamp("2031-06-26")
for a in ASSETS:
    d=pd.read_csv(f"{DATA}{a}.csv",parse_dates=["date"])
    d=d[d["date"]<=cur].set_index("date").sort_index()
    c=d["close"].astype(float)
    # last 60d vol
    r=c.pct_change()
    v=r.tail(60).std()
    # 20d return
    r20=(c.iloc[-1]/c.iloc[-21]-1) if len(c)>=21 else np.nan
    flat60 = (c.tail(60).diff().dropna().abs()<1e-9).mean() if len(c)>=60 else np.nan
    print(f"{a}: obs={len(c)} last={c.index[-1].date()} r20={r20:+.3f} vol60={v:.4f} flat60_frac={flat60:.2f}")
print("=== INDEX (obs-only) ===")
for f in ["DXY","USDCNY","USDJPY","EURUSD","VIX"]:
    d=pd.read_csv(f"{IDX}{f}.csv",parse_dates=["date"])
    d=d[d["date"]<=cur].set_index("date").sort_index()
    c=d["close"].astype(float)
    r20=(c.iloc[-1]/c.iloc[-21]-1) if len(c)>=21 else np.nan
    print(f"{f}: obs={len(c)} last={c.index[-1].date()} r20={r20:+.3f}")