"""Trader block review: 2034-08-03 -> 2034-08-17 per-asset returns."""
import sys
sys.path.insert(0, ".")

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
    t0 = df[df["date"].astype(str) >= "2034-08-03"]
    p0 = None
    if len(t0):
        p0 = t0.iloc[0]["close"]
    p1 = df.iloc[-1]["close"]
    d0 = df.iloc[-1]["date"]
    r = (p1 / p0 - 1.0) if p0 and p0 > 0 else None
    rows.append((s, p0, p1, r))

print(f"{'asset':<12}{'p0(08-03)':>12}{'p1':>12}{'ret%':>9}")
for s, p0, p1, r in rows:
    rr = f"{r*100:8.2f}" if r is not None else "     n/a"
    print(f"{s:<12}{str(p0):>12}{str(p1):>12}{rr:>9}")

start_nav = 1385498.22
end_nav = 1391426.01
print(f"\nNAV: {start_nav:.2f} -> {end_nav:.2f}  block pnl = {(end_nav/start_nav-1)*100:.2f}%")

# approximate attribution: weight at start * asset return
w0 = {"000300.SH": 0.0919, "SPX": 0.1594, "HSI": 0.0104, "N225": 0.0723,
      "SX5E": 0.0104, "000688.SH": 0.0699, "SOX": 0.0738, "NDX": 0.0970,
      "XAU": 0.1804, "COPPER": 0.1178, "WTI": 0.0427, "BTC": 0.0104,
      "ETH": 0.0428, "US10Y": 0.0104, "CN10Y": 0.0104}
print("\napprox attribution (w0 * ret):")
tot = 0.0
for s, p0, p1, r in rows:
    if r is None:
        continue
    c = w0.get(s, 0.0) * r * 100
    tot += c
    print(f"  {s:<12} {c:+7.3f}%")
print(f"  {'TOTAL':<12} {tot:+7.3f}%")
