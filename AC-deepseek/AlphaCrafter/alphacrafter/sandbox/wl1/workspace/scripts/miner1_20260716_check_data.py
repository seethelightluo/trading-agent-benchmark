"""Check data availability & volume for warm-up window <= 2026-07-15."""
import pandas as pd

CUT = pd.Timestamp("2026-07-15")
S = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
     "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

for s in S:
    d = pd.read_csv(f"../persistent/stock_data/{s}.csv")
    d['date'] = pd.to_datetime(d['date'])
    d = d[d['date'] <= CUT]
    print(f"{s:10s} rows={len(d):5d} vol_ok={d['volume'].notna().mean():.2f} "
          f"vol_nz={(d['volume'].fillna(0) > 0).mean():.2f} last={d['date'].max().date()} "
          f"first={d['date'].min().date()}")