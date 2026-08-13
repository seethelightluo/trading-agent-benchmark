"""Screener regime assessment - uses ONLY data through 2032-09-23 (previous completed trading day)."""
import pandas as pd
import numpy as np

CUTOFF = "2032-09-23"
assets = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

rows = []
for a in assets:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= CUTOFF].reset_index(drop=True)
    if len(df) < 70:
        print(a, "insufficient rows", len(df))
        continue
    px = df["close"]
    ret = px.pct_change()
    last = px.iloc[-1]
    ma20 = px.rolling(20).mean().iloc[-1]
    ma60 = px.rolling(60).mean().iloc[-1]
    r20 = last / px.iloc[-21] - 1
    r60 = last / px.iloc[-61] - 1
    r120 = last / px.iloc[-121] - 1 if len(df) > 121 else np.nan
    vol20 = ret.tail(20).std() * np.sqrt(252)
    vol60 = ret.tail(60).std() * np.sqrt(252)
    mean20 = ret.tail(20).mean()
    rows.append({
        "asset": a, "last": round(last,2), "r20": round(r20*100,2), "r60": round(r60*100,2),
        "r120": round(r120*100,2) if pd.notna(r120) else None,
        "above_ma20": last > ma20, "above_ma60": last > ma60,
        "vol20_ann": round(vol20*100,1), "vol60_ann": round(vol60*100,1),
        "mean20_daily": round(mean20*100,3)
    })

r = pd.DataFrame(rows)
print(r.to_string(index=False))
print()
print("Breadth above MA20:", int(r["above_ma20"].sum()), "/", len(r))
print("Breadth above MA60:", int(r["above_ma60"].sum()), "/", len(r))
print("20d cross-sectional dispersion (mean abs 20d ret):", round(r["r20"].abs().mean(),2), "%")
print("20d mean daily ret (equal-weight avg):", round(r["mean20_daily"].mean(),4), "%")
print("Median 20d ann vol:", round(r["vol20_ann"].median(),1), "%")
print("Max 20d ann vol:", r.loc[r["vol20_ann"].idxmax(),"asset"], r["vol20_ann"].max(), "%")
