"""Trader cycle60 (block 2029-10-18 -> 2029-11-01) per-asset return decomposition.
Uses block-start weights (10-18 close) and price return 10-18 -> 11-01 close.
Read-only: no account mutation.
"""
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

NAV0 = 1010953.559
NAV1 = 1003202.5858

# market values at block start (account state at 10-18 close, before step)
mv0 = {
    "000300.SH": 144394.0, "SPX": 86495.0, "HSI": 39120.0, "N225": 15733.0, "SX5E": 64258.0,
    "000688.SH": 129262.0, "SOX": 34819.0, "NDX": 29662.0, "XAU": 87373.0, "COPPER": 71859.0,
    "WTI": 27206.0, "BTC": 64258.0, "ETH": 82893.0, "US10Y": 66811.0, "CN10Y": 66811.0,
}

def ret(sym):
    df = get_stock_daily_data(symbol=sym, days=60)
    if df is None or len(df) < 20:
        df = get_index_daily_data(symbol=sym, days=60)
    if df is None or len(df) < 20:
        return None
    df = df.sort_values("date")
    dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
    c0 = c1 = None
    for i, d in enumerate(dates):
        if d == "2029-10-18":
            c0 = df.iloc[i]["close"]
        if d == "2029-11-01":
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
