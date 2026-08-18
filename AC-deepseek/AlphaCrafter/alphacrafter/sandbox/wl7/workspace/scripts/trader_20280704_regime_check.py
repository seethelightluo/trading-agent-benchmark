"""Trader 2028-07-04 regime snapshot at block end (visible thru 2028-07-03)."""
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

acct = get_account_dict()
assets = acct.get("watch_list", [])

def series(sym, days=90):
    try:
        df = get_index_daily_data(sym, days=days)
    except Exception:
        try:
            df = get_stock_daily_data(sym, days=days)
        except Exception:
            return None
    if df is None or "close" not in df or len(df) < 30:
        return None
    return pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))

# market (EW of 15 tradables)
rets = {}
for a in assets:
    s = series(a, 90)
    if s is not None:
        rets[a] = s.pct_change().rename(a)
R = pd.concat(rets, axis=1, join="inner").dropna()
mkt = R.mean(axis=1)
cp = (1.0 + R).cumprod()

def mkt_ret(n):
    return float(cp.iloc[-1] / cp.iloc[-1 - n] - 1.0) if len(cp) > n else float("nan")

print("last date:", R.index[-1].date(), " rows:", len(R))
for n in (5, 10, 20, 60):
    if len(R) > n:
        print(f"mkt{n}: {mkt_ret(n)*100:+.2f}%")

for name in ("VIX", "DXY", "USDJPY", "EURUSD", "USDCNY"):
    s = series(name, 90)
    if s is not None:
        r5 = s.iloc[-1] / s.iloc[-6] - 1 if len(s) > 6 else 0
        r20 = s.iloc[-1] / s.iloc[-21] - 1 if len(s) > 21 else 0
        print(f"{name}: last {s.iloc[-1]:.2f}  5d {r5*100:+.2f}%  20d {r20*100:+.2f}%")

# per-asset 20d & 60d returns
print("\nper-asset:")
for a in assets:
    s = series(a, 90)
    if s is None:
        print(f"  {a}: no data")
        continue
    r20 = s.iloc[-1] / s.iloc[-21] - 1 if len(s) > 21 else float("nan")
    r60 = s.iloc[-1] / s.iloc[-61] - 1 if len(s) > 61 else float("nan")
    print(f"  {a}: 20d {r20*100:+8.2f}%  60d {r60*100:+8.2f}%")
