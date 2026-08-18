"""Trader regime snapshot at 2028-06-19 (block end) for next-cycle context.
Read-only inspection of persistent data.
"""
import json, glob, os
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "persistent")
with open(os.path.join(BASE, "date.json")) as f:
    d = json.load(f)
print("date.json:", d if not isinstance(d, dict) else {k: str(v)[:30] for k, v in list(d.items())[:6]})

sdir = os.path.join(BASE, "stock_data")
rets = {}
for fpath in glob.glob(os.path.join(sdir, "*.csv")):
    sym = os.path.basename(fpath).replace(".csv", "")
    df = pd.read_csv(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    s = pd.Series(df["close"].astype(float).values, index=df["date"])
    rets[sym] = s.pct_change()
R = pd.concat(rets, axis=1, join="inner").dropna()
R = R[R.index <= pd.Timestamp("2028-06-19")]
mkt = R.mean(axis=1)
def cag(n):
    return (1.0 + mkt.tail(n)).prod() - 1.0
print(f"rows thru 06-19: {len(R)}  last date {R.index[-1]}")
print(f"mkt5  {cag(5)*100:+.2f}%  mkt10 {cag(10)*100:+.2f}%  mkt20 {cag(20)*100:+.2f}%  mkt60 {cag(min(60,len(R)))*100:+.2f}%")

last20 = R.tail(20)
base20 = R.tail(21).iloc[0]
for sym in sorted(R.columns):
    r20 = (last20[sym].iloc[-1] / base20[sym] - 1) * 100
    print(f"  {sym:10s} 20d {r20:+7.2f}%")

for sig in ("VIX", "DXY", "USDJPY", "EURUSD", "USDCNY"):
    fp = os.path.join(BASE, "index_data", sig + ".csv")
    if not os.path.exists(fp):
        continue
    df = pd.read_csv(fp)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df[df["date"] <= pd.Timestamp("2028-06-19")]
    c = df["close"].astype(float)
    r5 = (c.iloc[-1] / c.iloc[-6] - 1) * 100 if len(c) > 6 else float("nan")
    r20 = (c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(c) > 21 else float("nan")
    print(f"{sig:8s} last {c.iloc[-1]:10.2f}  5d {r5:+6.2f}%  20d {r20:+6.2f}%")
