"""Trader cycle analysis 2030-02-21 -> 2030-03-07: per-asset PnL contributions."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

# start-of-block market values (from account.json at 2030-02-21)
start_mv = {
    "000300.SH": 117461.7, "SPX": 132330.2, "HSI": 8290.7, "N225": 122735.8,
    "SX5E": 8290.7, "000688.SH": 45482.2, "SOX": 45466.0, "NDX": 98651.4,
    "XAU": 161726.0, "COPPER": 32423.8, "WTI": 68415.3, "BTC": 8290.7,
    "ETH": 59360.3, "US10Y": 8290.7, "CN10Y": 8290.7,
}
nav0 = 925506.19

def get_df(sym):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=30)
        return get_stock_daily_data(sym, days=30)
    except Exception:
        return None

rows = []
for a in ASSETS:
    df = get_df(a)
    if df is None or len(df) < 10:
        rows.append((a, None, None, None))
        continue
    df = df.sort_values("date")
    dts = [str(x)[:10] for x in df["date"]]
    # find close at 2030-02-20 (block start reference) and 2030-03-06 (last full day)
    def close_on(target):
        for i in range(len(df) - 1, -1, -1):
            if dts[i] <= target:
                return float(df["close"].iloc[i]), dts[i]
        return None, None
    c0, d0 = close_on("2030-02-20")
    c1, d1 = close_on("2030-03-06")
    if c0 and c1:
        ret = c1 / c0 - 1.0
        pnl = start_mv[a] * ret
        rows.append((a, ret, pnl, d1))
    else:
        rows.append((a, None, None, None))

print(f"{'asset':10s} {'ret%':>8s} {'pnl_est':>10s} {'last_date':>12s}")
total = 0.0
for a, ret, pnl, d1 in rows:
    if ret is None:
        print(f"{a:10s} {'n/a':>8s}")
        continue
    total += pnl
    print(f"{a:10s} {ret*100:8.2f} {pnl:10.0f} {str(d1):>12s}")
print(f"{'TOTAL':10s} {'':>8s} {total:10.0f}")
print("actual net change:", 927552.47 - nav0)
