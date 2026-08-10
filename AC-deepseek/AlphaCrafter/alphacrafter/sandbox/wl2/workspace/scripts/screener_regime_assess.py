"""Screener: market regime assessment from raw CSVs, truncated at runtime current date."""
import os, numpy as np, pandas as pd, json

CUR = pd.Timestamp("2026-07-30")  # runtime current date; use data through previous close
DATA = "../persistent/stock_data"
IDX = "../persistent/index_data"
TRADABLE = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX",
            "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
OBS = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def load(p):
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    dc = "date" if "date" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc])
    df = df.sort_values(dc).reset_index(drop=True)
    df = df[df[dc] < CUR]  # no lookahead
    return df, dc

rows = []
for s in TRADABLE:
    df, dc = load(os.path.join(DATA, s + ".csv"))
    c = df["close"] if "close" in df.columns else df[df.columns[1]]
    last = df[dc].iloc[-1]
    ret = {}
    for w in (20, 60, 120, 250):
        ret[w] = c.iloc[-1] / c.iloc[-1 - w] - 1.0 if len(c) > w else np.nan
    r = c.pct_change()
    vol20 = r.tail(20).std() * np.sqrt(252) if len(r) >= 21 else np.nan
    ma60 = c.rolling(60).mean().iloc[-1]
    rows.append(dict(asset=s, last=str(last.date()), r20=ret[20], r60=ret[60],
                     r120=ret[120], r250=ret[250], vol20=vol20, above60=c.iloc[-1] > ma60))

tab = pd.DataFrame(rows).set_index("asset")
print("=== last trading date (truncated):", tab["last"].max(), "===")
print(tab[["last","r20","r60","r120","r250","vol20","above60"]].to_string(float_format=lambda x: f"{x:.4f}"))
print("\nbreadth above 60d MA: %d/15" % int(tab["above60"].sum()))
print("median 20d vol (ann): %.3f | max: %.3f (%s)" % (tab["vol20"].median(), tab["vol20"].max(), tab["vol20"].idxmax()))

rets = {}
for s in TRADABLE:
    df, dc = load(os.path.join(DATA, s + ".csv"))
    c = df["close"] if "close" in df.columns else df[df.columns[1]]
    rets[s] = c.pct_change().tail(20).reset_index(drop=True)
R = pd.DataFrame(rets).dropna()
corr = R.corr()
vals = corr.values[np.triu_indices(len(corr), 1)]
print("mean pairwise |20d-return corr|: %.3f" % np.abs(vals).mean())
print("cross-sectional dispersion (ann): %.3f" % (R.std(axis=0).mean() * np.sqrt(252)))

for s in OBS:
    df, dc = load(os.path.join(IDX, s + ".csv"))
    c = df["close"] if "close" in df.columns else df[df.columns[1]]
    r20 = c.iloc[-1]/c.iloc[-21]-1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1]/c.iloc[-61]-1 if len(c) > 61 else np.nan
    print(f"OBS {s}: last={df[dc].iloc[-1].date()} close={c.iloc[-1]:.3f} r20={r20:.4f} r60={r60:.4f}")
