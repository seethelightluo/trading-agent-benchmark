"""Compute per-asset returns over the live block 2027-12-30 -> 2028-01-13,
plus regime drift at the decision date, using only data visible at each date."""
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

WATCH = get_account_dict()["watch_list"]


def series(a):
    for fn in (get_stock_daily_data, get_index_daily_data):
        try:
            df = fn(a, days=300)
            if df is not None and len(df):
                return df
        except Exception:
            continue
    return None


print("=== per-asset block returns (close-to-close) ===")
block_ret = {}
for a in WATCH:
    df = series(a)
    if df is None or len(df) < 30:
        print(f"{a:10s} no data")
        continue
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    # last completed bar of the block = last row; decision/start = 2027-12-30
    try:
        idx_start = dates[dates <= pd.Timestamp("2027-12-30")].index[-1]
    except IndexError:
        print(f"{a:10s} start date missing")
        continue
    p_start = float(df.loc[idx_start, "close"])
    p_end = float(df.iloc[-1]["close"])
    r = p_end / p_start - 1.0
    block_ret[a] = r
    print(f"{a:10s} start={p_start:12.4f} end={p_end:12.4f} block_ret={r*100:7.2f}%")

print("\n=== regime drift at 2027-12-30 decision (cross-asset 20d t-stat) ===")
closes = {}
for a in WATCH:
    df = series(a)
    if df is None:
        continue
    df = df.sort_values("date").reset_index(drop=True)
    s = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
    closes[a] = s
panel = pd.concat(closes, axis=1).dropna()
pre = panel[panel.index <= pd.Timestamp("2027-12-30")]
if len(pre) >= 30:
    rets = pre.pct_change().dropna()
    mkt = rets.mean(axis=1)
    r20 = float(mkt.tail(20).mean())
    v20 = float(mkt.tail(20).std())
    trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
    regime = "bull" if trend > 1.0 else ("bear" if trend < -1.0 else "sideways")
    print(f"trend_t={trend:.3f} regime={regime}")

print("\n=== flat-series check (60d return) ===")
for a in ["HSI", "000688.SH", "CN10Y", "US10Y"]:
    df = series(a)
    if df is None or len(df) < 65:
        continue
    df = df.sort_values("date").reset_index(drop=True)
    r60 = float(df.iloc[-1]["close"]) / float(df.iloc[-61]["close"]) - 1.0
    print(f"{a:10s} 60d_ret={r60*100:6.3f}%")
