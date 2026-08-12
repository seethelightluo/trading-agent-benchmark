"""Cycle 48 attribution: price returns over block 02-22 -> 03-08 (decision 02-21 -> visible 03-07)."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

acc = get_account_dict()
na = acc["net_assets"]
w_end = {p["symbol"]: p["market_value"]/na for p in acc["positions"]}

def get(sym, days=40):
    try:
        df = get_stock_daily_data(sym, days=days)
    except Exception:
        df = None
    if df is None:
        try:
            df = get_index_daily_data(sym, days=days)
        except Exception:
            return None
    return df

rows = []
for s in WATCH:
    df = get(s)
    if df is None or len(df) < 25:
        rows.append((s, float('nan'), w_end.get(s, 0)))
        continue
    df = df.sort_values("date")
    c = df["close"].astype(float)
    start = float(c.iloc[-11])   # close 10 trading days before last (block start ~02-21)
    end = float(c.iloc[-1])      # visible through 03-07
    ret = end/start - 1.0
    rows.append((s, ret, w_end.get(s, 0)))

rows.sort(key=lambda x: -x[1] if x[1] == x[1] else 1e9)
print(f"{'sym':10s} {'block_ret':>9s} {'w_end':>7s}")
for s, r, we in rows:
    print(f"{s:10s} {r*100:8.2f}% {we*100:6.2f}%")
