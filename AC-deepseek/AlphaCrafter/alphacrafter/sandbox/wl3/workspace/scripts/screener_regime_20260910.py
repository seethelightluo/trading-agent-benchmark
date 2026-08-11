"""Regime assessment for factor screening on 2026-09-10 (data through 2026-09-09)."""
import pandas as pd
import numpy as np

CUR = "2026-09-10"
LAST = "2026-09-09"  # previous completed trading day

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

px = {}
for a in ASSETS:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
    df = df[df["date"] <= LAST].set_index("date")["close"].astype(float)
    px[a] = df

panel = pd.DataFrame(px).dropna()
print("panel rows:", len(panel), "| last:", panel.index[-1].date())

ret = panel.pct_change().dropna()

# --- per-asset stats ---
print("\n=== PER-ASSET (through 2026-09-09) ===")
row = {}
for a in ASSETS:
    s = px[a]
    r = s.pct_change().dropna()
    ret20 = s.iloc[-1] / s.iloc[-21] - 1 if len(s) > 21 else np.nan
    ret60 = s.iloc[-1] / s.iloc[-61] - 1 if len(s) > 61 else np.nan
    vol20 = r.tail(20).std() * np.sqrt(252)
    above20 = s.iloc[-1] > s.tail(21).mean()
    above60 = s.iloc[-1] > s.tail(61).mean()
    row[a] = (ret20, ret60, vol20)
    print(f"{a:10s} ret20={ret20*100:7.2f}%  ret60={ret60*100:7.2f}%  vol20={vol20*100:5.1f}%  >MA20={above20} >MA60={above60}")

rets20 = {a: row[a][0] for a in ASSETS}
rets60 = {a: row[a][1] for a in ASSETS}
print("\n=== CROSS-SECTIONAL DISPERSION (20d) ===")
print("mean ret20:", np.mean(list(rets20.values())) * 100, "%")
print("median ret20:", np.median(list(rets20.values())) * 100, "%")
print("dispersion (std of ret20):", np.std(list(rets20.values())) * 100, "%")
print("assets up 20d:", sum(1 for v in rets20.values() if v > 0), "/", len(ASSETS))
print("assets up 60d:", sum(1 for v in rets60.values() if v > 0), "/", len(ASSETS))

# --- equal-weight portfolio regime ---
ew = panel.pct_change().dropna().mean(axis=1)
wealth = (1 + ew).cumprod()
mdd = (wealth / wealth.rolling(60).max() - 1).min()
print("\n=== EW PORTFOLIO ===")
print("since online (07-16) cumulative:", wealth.iloc[-1] / wealth.loc["2026-07-16"] - 1)
print("60d max drawdown:", mdd)
print("20d mean daily ret:", ew.tail(20).mean())
print("20d ann vol:", ew.tail(20).std() * np.sqrt(252))

# --- correlation regime (20d) ---
c = ret.tail(20).corr()
vals = c.values[np.triu_indices(len(c), k=1)]
print("\navg pairwise corr (20d):", np.mean(vals), "| median:", np.median(vals))

# --- observation signals ---
for sym in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    df = pd.read_csv(f"../persistent/index_data/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= LAST].set_index("date")["close"].astype(float)
    r5 = df.iloc[-1] / df.iloc[-6] - 1 if len(df) > 6 else np.nan
    r20 = df.iloc[-1] / df.iloc[-21] - 1 if len(df) > 21 else np.nan
    r60 = df.iloc[-1] / df.iloc[-61] - 1 if len(df) > 61 else np.nan
    print(f"{sym:8s} last={df.iloc[-1]:10.2f}  r5={r5*100:6.2f}%  r20={r20*100:6.2f}%  r60={r60*100:6.2f}%")

# --- trend strength: fraction of up days, consecutive direction ---
ew20 = ew.tail(20)
print("\nEW up-day ratio (20d):", (ew20 > 0).mean())
print("last 5 EW daily rets:", [round(x * 100, 2) for x in ew.tail(5)])
