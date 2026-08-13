"""Screener regime assessment -- data through visible_through (2030-11-13).
READ-ONLY analysis: no account/date mutation, no backtest/step.
"""
import pandas as pd
import numpy as np

CUT = "2030-11-13"
ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def load(sym, obs=False):
    p = f"../persistent/index_data/{sym}.csv" if obs else f"../persistent/stock_data/{sym}.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= CUT].set_index("date").sort_index()
    return df["close"].astype(float)

closes = {a: load(a) for a in ASSETS}
obs = {s: load(s, obs=True) for s in ["VIX","DXY","USDJPY","USDCNY","EURUSD"]}

px = pd.DataFrame(closes)
rets = px.pct_change()

print("=== LEVELS @", CUT, "===")
for a in ASSETS:
    print(f"{a:10s} {px[a].iloc[-1]:12.2f}")

print("\n=== RETURNS (%) ===")
print(f"{'asset':10s} {'5d':>8s} {'10d':>8s} {'20d':>8s} {'60d':>8s} {'120d':>8s} {'180d':>8s}")
for a in ASSETS:
    row = []
    for h in [5, 10, 20, 60, 120, 180]:
        if len(px[a]) > h:
            r = (px[a].iloc[-1] / px[a].iloc[-1 - h] - 1) * 100
        else:
            r = np.nan
        row.append(r)
    print(f"{a:10s} " + " ".join(f"{v:8.2f}" for v in row))

print("\n=== OBSERVATION SIGNALS ===")
for s, c in obs.items():
    last = c.iloc[-1]
    r5 = (c.iloc[-1]/c.iloc[-6]-1)*100 if len(c) > 5 else np.nan
    r20 = (c.iloc[-1]/c.iloc[-21]-1)*100 if len(c) > 20 else np.nan
    print(f"{s:8s} {last:10.2f}  5d {r5:7.2f}%  20d {r20:7.2f}%")

print("\n=== VOLATILITY (20d ann, %) ===")
for a in ASSETS:
    v = rets[a].tail(20).std() * np.sqrt(252) * 100
    print(f"{a:10s} {v:8.2f}")

print("\n=== CROSS-SECTION ===")
r20 = rets.tail(20)
print("mean pairwise corr (20d):", round(r20.corr().values[np.triu_indices(15,1)].mean(), 4))
print("avg daily |ret| 20d (%):", round(r20.abs().mean().mean()*100, 3))
print("dispersion 20d (std of asset 20d rets, %):", round(rets.tail(20).mean().std()*100, 2))

# trend proxies
mkt = rets.mean(axis=1)
print("\n=== MARKET (equal-weight) ===")
for h in [5,10,20,60]:
    print(f"{h}d: {(1+mkt.tail(h)).prod()-1:.4f}" if len(mkt)>=h else "")

# VIX regime
vix = obs["VIX"]
print("\n=== VIX ===")
print("last:", round(vix.iloc[-1],2), "| 5d ago:", round(vix.iloc[-6],2), "| 20d ago:", round(vix.iloc[-21],2) if len(vix)>20 else None)
print("VIX 5d chg %:", round((vix.iloc[-1]/vix.iloc[-6]-1)*100,1))
