from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def fetch(sym):
    try:
        return get_stock_daily_data(sym, days=20)
    except Exception:
        return None


rows = []
for s in WATCH:
    df = fetch(s)
    if df is None or len(df) < 12:
        rows.append((s, None, None))
        continue
    df = df.sort_values("date").reset_index(drop=True)
    # block: last 10 trading days (02-01 .. 02-15), return from close[-11] to close[-1]
    c = df["close"].astype(float)
    ret_10d = c.iloc[-1] / c.iloc[-11] - 1.0
    ret_1d = c.iloc[-1] / c.iloc[-2] - 1.0
    rows.append((s, ret_10d, ret_1d))

rows.sort(key=lambda r: -(r[1] if r[1] is not None else -9))
for s, r10, r1 in rows:
    print(f"{s:10s} block_ret={r10*100:+.2f}%  last_day={r1*100:+.2f}%" if r10 is not None else f"{s:10s} NA")
