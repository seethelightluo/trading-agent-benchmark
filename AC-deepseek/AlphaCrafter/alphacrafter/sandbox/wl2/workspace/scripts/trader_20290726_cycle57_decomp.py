"""Trader cycle57 (block 2029-07-26 -> 2029-08-09) per-asset return decomposition.
Uses data visible at decision/end of block (07-25 close -> 08-08 close).
Read-only: no account mutation.
"""
import json
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

NAV0 = 1016877.2649
NAV1 = 1011610.31

# positions at block start (from account before step, 07-25 close)
pos0 = {
    "000300.SH": 4.7, "SPX": 19.23, "HSI": 1.71, "N225": 1.17, "SX5E": 10.06,
    "000688.SH": 19.4, "SOX": 2.37, "NDX": 2.23, "XAU": 36.15, "COPPER": 4568.11,
    "WTI": 356.08, "BTC": 0.99, "ETH": 30.85, "US10Y": 13935.83, "CN10Y": 36523.91,
}
# prices at block start (mv/qty from account before step)
mv0 = {
    "000300.SH": 20063.0, "SPX": 143573.0, "HSI": 43683.0, "N225": 95506.0, "SX5E": 63238.0,
    "000688.SH": 64296.0, "SOX": 20544.0, "NDX": 56970.0, "XAU": 147903.0, "COPPER": 23571.0,
    "WTI": 24164.0, "BTC": 63238.0, "ETH": 122755.0, "US10Y": 63687.0, "CN10Y": 63687.0,
}

def ret(sym):
    df = get_stock_daily_data(symbol=sym, days=60)
    if df is None or len(df) < 20:
        df = get_index_daily_data(symbol=sym, days=60)
    if df is None or len(df) < 20:
        return None
    df = df.sort_values("date")
    dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
    # find 07-25 and 08-08 close
    c0 = c1 = None
    for i, d in enumerate(dates):
        if d == "2029-07-25":
            c0 = df.iloc[i]["close"]
        if d == "2029-08-08":
            c1 = df.iloc[i]["close"]
    if c0 is None or c1 is None or c0 == 0:
        return None
    return c1 / c0 - 1.0, dates[0], dates[-1]

tot = 0.0
print(f"{'sym':10s} {'w0':>7s} {'ret%':>8s} {'contrib%':>9s}")
rows = []
for sym, mv in mv0.items():
    w = mv / NAV0
    r = ret(sym)
    if r is None:
        print(f"{sym:10s} {w*100:6.2f}%  n/a")
        continue
    rv, d0, d1 = r
    contrib = w * rv * 100
    tot += contrib
    rows.append((sym, w, rv, contrib))
    print(f"{sym:10s} {w*100:6.2f}% {rv*100:7.2f}% {contrib:8.3f}%  ({d0}->{d1})")
print(f"\nsum approx contrib: {tot:.3f}% | actual period ret: {(NAV1/NAV0-1)*100:.3f}%")
