"""Screener: assess market regime as of 2027-06-02 (visible through)."""
import pandas as pd, numpy as np, json, os

DATA = "../persistent/stock_data"
MACRO = "../persistent/index_data"
WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO_S = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def load(sym):
    df = pd.read_csv(os.path.join(DATA, sym + ".csv"), parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df

series = {s: load(s) for s in WATCH}
macro = {}
for s in MACRO_S:
    p = os.path.join(MACRO, s + ".csv")
    if os.path.exists(p):
        m = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
        macro[s] = m

# Restrict to visible through 2027-06-02
cut = "2027-06-02"
for s, df in series.items():
    series[s] = df[df.index <= cut]
for s, m in macro.items():
    macro[s] = m[m.index <= cut]

print("=== last close dates ===")
for s, df in series.items():
    print(s, df.index[-1].date(), round(df["close"].iloc[-1], 2))

# returns
print("\n=== returns (pct) ===")
out = {}
for s, df in series.items():
    c = df["close"]
    r5 = c.iloc[-1]/c.iloc[-6]-1 if len(c) > 6 else np.nan
    r20 = c.iloc[-1]/c.iloc[-21]-1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1]/c.iloc[-61]-1 if len(c) > 61 else np.nan
    r120 = c.iloc[-1]/c.iloc[-121]-1 if len(c) > 121 else np.nan
    # ann vol 20d
    rets = df["close"].pct_change().dropna()
    v20 = rets.iloc[-20:].std()*np.sqrt(252) if len(rets) >= 20 else np.nan
    out[s] = dict(r5=r5*100, r20=r20*100, r60=r60*100, r120=r120*100, vol20=v20*100)
    print(f"{s:12s} 5d {r5*100:7.2f}%  20d {r20*100:7.2f}%  60d {r60*100:7.2f}%  120d {r120*100:7.2f}%  vol20 {v20*100:6.1f}%")

print("\n=== macro ===")
for s, m in macro.items():
    c = m["close"] if "close" in m.columns else m.iloc[:, 1]
    c = pd.to_numeric(c, errors="coerce").dropna()
    if len(c) > 21:
        print(f"{s:8s} last {c.iloc[-1]:.2f}  20d {c.iloc[-1]/c.iloc[-21]-1:+.2%}  60d {c.iloc[-1]/c.iloc[-61]-1:+.2%}" if len(c)>61 else f"{s} last {c.iloc[-1]:.2f} 20d {c.iloc[-1]/c.iloc[-21]-1:+.2%}")

# VIX level
if "VIX" in macro:
    v = pd.to_numeric(macro["VIX"]["close"], errors="coerce").dropna()
    print("\nVIX last:", v.iloc[-1], "20d ago:", v.iloc[-21] if len(v) > 21 else None, "60d ago:", v.iloc[-61] if len(v) > 61 else None)
    print("VIX min/max last 120d:", v.iloc[-120:].min(), v.iloc[-120:].max())

# MA structure
print("\n=== MA structure (close vs MA20/MA60) ===")
for s, df in series.items():
    c = df["close"]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    above = c.iloc[-1] > ma20
    ma20_above_ma60 = ma20 > ma60
    print(f"{s:12s} close {c.iloc[-1]:9.2f} MA20 {ma20:9.2f} MA60 {ma60:9.2f} close>MA20 {above} MA20>MA60 {ma20_above_ma60}")

# cross-sectional dispersion
print("\n=== cross-sectional dispersion (20d) ===")
r20s = [out[s]["r20"] for s in WATCH]
print("mean 20d:", np.mean(r20s), "std:", np.std(r20s), "max:", max(r20s), "min:", min(r20s))

# save snapshot
json.dump(out, open("scripts/_regime_snapshot_20270603.json", "w"), indent=1, default=str)
