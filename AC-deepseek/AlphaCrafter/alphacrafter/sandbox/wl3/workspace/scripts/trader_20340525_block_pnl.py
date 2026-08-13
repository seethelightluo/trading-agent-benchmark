import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
assets = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
BLOCK_START = "2034-05-11"


def get_df(sym, days=40):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=days)
        return get_stock_daily_data(sym, days=days)
    except Exception:
        return None


rows = []
for a in assets:
    df = get_df(a, 40)
    if df is None or len(df) < 15:
        print(a, "NO DATA")
        continue
    df = df.sort_values("date").reset_index(drop=True)
    m = df[df["date"].astype(str).str[:10] == BLOCK_START]
    if len(m) == 0:
        print(a, "block start not in window")
        continue
    i0 = m.index[0]
    p0 = df.loc[i0, "close"]
    p1 = df.iloc[-1]["close"]
    rows.append((a, 100 * (p1 / p0 - 1)))

rows.sort(key=lambda x: -x[1])
for a, r in rows:
    print(f"{a:10s} {r:+8.2f}%")
