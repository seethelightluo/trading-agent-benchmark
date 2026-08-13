"""Trader block review: 2034-07-20 -> 2034-08-03.

Compute per-asset block returns (visible through 2034-08-02) and rough PnL
attribution using the account state at block start vs end.
"""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

def get_df(symbol, days=40):
    try:
        if symbol in OBS:
            return get_index_daily_data(symbol, days=days)
        return get_stock_daily_data(symbol, days=days)
    except Exception:
        return None

watch = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

rows = []
for s in watch:
    df = get_df(s, days=40)
    if df is None or len(df) < 12:
        rows.append((s, None, None, None))
        continue
    df = df.sort_values("date").reset_index(drop=True)
    # find 2034-07-20 close and last close
    t0 = df[df["date"].astype(str) >= "2034-07-20"]
    p0 = None
    if len(t0):
        p0 = t0.iloc[0]["close"]
    p1 = df.iloc[-1]["close"]
    d0 = df.iloc[-1]["date"]
    r = (p1 / p0 - 1.0) if p0 and p0 > 0 else None
    rows.append((s, p0, p1, r))

print(f"{'asset':<12}{'p0(07-20)':>12}{'p1':>12}{'ret%':>9}")
for s, p0, p1, r in rows:
    rr = f"{r*100:8.2f}" if r is not None else "     n/a"
    print(f"{s:<12}{str(p0):>12}{str(p1):>12}{rr:>9}")

# Account start/end NAV
start_nav = 1403539.15
end_nav = 1385498.22
print(f"\nNAV: {start_nav:.2f} -> {end_nav:.2f}  block pnl = {(end_nav/start_nav-1)*100:.2f}%")
