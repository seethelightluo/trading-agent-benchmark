"""Trader attribution for block 2035-08-16 -> 2035-08-30."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

rows = []
for s in WATCH:
    try:
        df = get_stock_daily_data(symbol=s, days=30)
    except Exception:
        df = None
    if df is None or len(df) < 2:
        rows.append((s, None))
        continue
    df = df.sort_values("date")
    dates = [str(x)[:10] for x in df["date"]]
    # find 08-16 and 08-30 closes
    c = dict(zip(dates, df["close"].astype(float)))
    p0 = c.get("2035-08-16")
    p1 = c.get("2035-08-30")
    if p0 is None or p1 is None:
        # use last two available
        p0, p1 = float(df["close"].iloc[-2]), float(df["close"].iloc[-1])
    rows.append((s, (p1 / p0 - 1.0) * 100.0 if p0 else None))

for s, r in sorted(rows, key=lambda x: -(x[1] or -999)):
    print(f"{s:12s} {r if r is None else round(r,2):>8}%")
