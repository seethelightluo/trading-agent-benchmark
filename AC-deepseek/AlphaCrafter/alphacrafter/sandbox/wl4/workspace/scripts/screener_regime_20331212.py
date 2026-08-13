"""Screener regime assessment - 2033-12-12 cycle. Uses ONLY data through visible_through 2033-12-09."""
import pandas as pd, numpy as np, glob, json

CUTOFF = "2033-12-09"
assets = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
macro = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

closes = {}
rets = {}
for a in assets:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df = df[df["date"] <= CUTOFF].reset_index(drop=True)
    closes[a] = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
    rets[a] = closes[a].pct_change()

px = pd.DataFrame(closes)
r = pd.DataFrame(rets)

def ann_vol(s):
    return s.std() * np.sqrt(252)

print("="*100)
print("REGIME ASSESSMENT through", CUTOFF, " (last obs:", px.index.max().date(), ")")
print("="*100)
rows = []
for a in assets:
    c = px[a]
    rr = r[a]
    last = c.iloc[-1]
    ret21 = c.iloc[-1]/c.iloc[-22]-1 if len(c) > 22 else np.nan
    ret63 = c.iloc[-1]/c.iloc[-64]-1 if len(c) > 64 else np.nan
    ret126 = c.iloc[-1]/c.iloc[-127]-1 if len(c) > 127 else np.nan
    ret10 = c.iloc[-1]/c.iloc[-11]-1 if len(c) > 11 else np.nan
    vol21 = ann_vol(rr.iloc[-21:])
    vol63 = ann_vol(rr.iloc[-63:])
    # MA slope over last 21d: (MA20 today - MA20 21d ago)/MA20 21d ago
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    ma20_slope = ma20.iloc[-1]/ma20.iloc[-22]-1 if len(ma20.dropna()) > 21 else np.nan
    # drawdown from 126d high
    dd = c.iloc[-1]/c.iloc[-126:].max()-1 if len(c) >= 126 else np.nan
    rows.append([a, round(ret10*100,1), round(ret21*100,1), round(ret63*100,1), round(ret126*100,1),
                 round(vol21*100,1), round(vol63*100,1), round(ma20_slope*100,1), round(dd*100,1)])
res = pd.DataFrame(rows, columns=["asset","r10d%","r21d%","r63d%","r126d%","vol21%","vol63%","ma20slope%","dd126%"])
print(res.to_string(index=False))

print()
print("="*100)
print("MACRO OBSERVATION-ONLY (through", CUTOFF, ")")
print("="*100)
for m in macro:
    df = pd.read_csv(f"../persistent/index_data/{m}.csv")
    df = df[df["date"] <= CUTOFF].reset_index(drop=True)
    c = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
    rr = c.pct_change()
    ret21 = c.iloc[-1]/c.iloc[-22]-1 if len(c) > 22 else np.nan
    ret63 = c.iloc[-1]/c.iloc[-64]-1 if len(c) > 64 else np.nan
    vol21 = ann_vol(rr.iloc[-21:])
    print(f"{m:8s} last={c.iloc[-1]:10.2f} 21d={ret21*100:7.2f}% 63d={ret63*100:7.2f}% vol21={vol21*100:5.1f}%")

# Cross-asset correlation regime
print()
print("="*100)
print("CROSS-ASSET CORRELATION (63d, last 63 obs)")
print("="*100)
corr63 = r.iloc[-63:].corr()
avg_corr = (corr63.values[np.triu_indices(15, k=1)]).mean()
print(f"avg pairwise corr (63d): {avg_corr:.3f}")
corr21 = r.iloc[-21:].corr()
print(f"avg pairwise corr (21d): {(corr21.values[np.triu_indices(15, k=1)]).mean():.3f}")

# momentum breadth: how many assets positive 63d
print()
print("Breadth 63d positive:", int((res['r63d%'] > 0).sum()), "/ 15 ; 21d positive:", int((res['r21d%'] > 0).sum()), "/ 15")

# dispersion
disp63 = r.iloc[-63:].mean(axis=1).std()*np.sqrt(252)
print(f"dispersion (std of equal-weight mkt ret, ann): {disp63*100:.2f}%")

# equity trend: SPX & 000300
print()
print("Trend proxies: SPX ma20slope", round(res.loc[res.asset=='SPX','ma20slope%'].iloc[0],2),
      " 000300 ma20slope", round(res.loc[res.asset=='000300.SH','ma20slope%'].iloc[0],2))
