"""Regime assessment for factor screening cycle 2026-07-30."""
import pandas as pd
import numpy as np
import glob

CUR = pd.Timestamp("2026-07-30")
VIS = pd.Timestamp("2026-07-29")  # visible through previous completed day

ASSETS = ["000300.SH", "000688.SH", "SPX", "NDX", "SOX", "HSI", "N225", "SX5E",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]


def load(sym, folder="../persistent/stock_data/"):
    df = pd.read_csv(f"{folder}{sym}.csv", parse_dates=["date"]).sort_values("date")
    df = df[df["date"] <= VIS].set_index("date")
    return df


px = {}
for s in ASSETS + MACRO:
    folder = "../persistent/index_data/" if s in MACRO else "../persistent/stock_data/"
    px[s] = load(s, folder)["close"]

pxdf = pd.DataFrame(px).dropna(how="all")
print("Data range:", pxdf.index.min().date(), "->", pxdf.index.max().date(), "| rows:", len(pxdf))

ret = pxdf.pct_change().dropna()
last = pxdf.index[-1]

print("\n=== Recent returns & trend (through %s) ===" % last.date())
rows = []
for s in ASSETS + MACRO:
    x = pxdf[s].dropna()
    if len(x) < 30:
        continue
    r1m = x.iloc[-1] / x.iloc[-22] - 1 if len(x) > 22 else np.nan
    r3m = x.iloc[-1] / x.iloc[-66] - 1 if len(x) > 66 else np.nan
    r6m = x.iloc[-1] / x.iloc[-132] - 1 if len(x) > 132 else np.nan
    sma20 = x.iloc[-20:].mean()
    sma60 = x.iloc[-60:].mean()
    vol20 = ret[s].iloc[-20:].std() * np.sqrt(252) if len(ret[s].dropna()) >= 20 else np.nan
    vol60 = ret[s].iloc[-60:].std() * np.sqrt(252) if len(ret[s].dropna()) >= 60 else np.nan
    rows.append([s, r1m, r3m, r6m, vol20, vol60, x.iloc[-1] / sma20 - 1, x.iloc[-1] / sma60 - 1])

out = pd.DataFrame(rows, columns=["sym", "ret_1m", "ret_3m", "ret_6m", "vol20_ann", "vol60_ann", "vs_sma20", "vs_sma60"])
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))

# Market-wide stats on tradable universe
tr = ret[ASSETS]
print("\n=== Cross-asset stats (tradable universe) ===")
print("Avg 20d realized vol (ann):", tr.iloc[-20:].std().mean() * np.sqrt(252))
print("Cross-sectional dispersion 20d (mean abs daily ret):", tr.iloc[-20:].abs().mean().mean())
corr = tr.iloc[-60:].corr()
print("Avg pairwise |corr| 60d:", (corr.abs().values[np.triu_indices(len(ASSETS), 1)]).mean())
print("Median pairwise corr 60d:", np.median(corr.values[np.triu_indices(len(ASSETS), 1)]))

# Macro regime
print("\n=== Macro signals ===")
for s in ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]:
    x = pxdf[s].dropna()
    if len(x) < 30:
        continue
    r1m = x.iloc[-1] / x.iloc[-22] - 1
    r3m = x.iloc[-1] / x.iloc[-66] - 1
    print(f"{s:8s} last={x.iloc[-1]:9.2f}  1m={r1m*100:6.2f}%  3m={r3m*100:6.2f}%  vs_sma60={x.iloc[-1]/x.iloc[-60:].mean()-1:+.3%}")

# Recent trend strength: count of up days in last 60 for SPX / BTC / HSI
for s in ["SPX", "BTC", "HSI", "NDX", "XAU", "WTI"]:
    r = ret[s].dropna().iloc[-60:]
    print(f"{s:6s} 60d up-day ratio: {(r > 0).mean():.2f}  60d cum ret: {(1+r).prod()-1:+.1%}")
