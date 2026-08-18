from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd

SYMS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
        "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

acc = get_account_dict()
mv = {p["symbol"]: p["market_value"] for p in acc["positions"]}
total = acc["total_assets"]

def get_df(sym, days=40):
    df = get_stock_daily_data(symbol=sym, days=days)
    if df is None:
        df = get_index_daily_data(symbol=sym, days=days)
    if df is None:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    return df

rows = []
for s in SYMS:
    df = get_df(s)
    if df is None or len(df) < 15:
        print(s, "NO DATA", None if df is None else len(df))
        continue
    # last completed day of prev block = 2029-05-02; block end mark = last row
    base = df[df["date"] <= pd.Timestamp("2029-05-02")]
    if len(base) == 0:
        base = df.iloc[:1]
    p0 = base.iloc[-1]["close"]
    p1 = df.iloc[-1]["close"]
    ret = (p1 - p0) / p0
    w = mv.get(s, 0.0) / total
    rows.append((s, p0, p1, ret * 100, w, ret * w * 100))

rows.sort(key=lambda r: r[5], reverse=True)
print(f"{'sym':9s} {'p0':>10s} {'p1':>10s} {'ret%':>8s} {'w':>7s} {'contrib%':>9s}")
tot_contrib = 0.0
for s, p0, p1, ret, w, c in rows:
    print(f"{s:9s} {p0:10.4f} {p1:10.4f} {ret:8.2f} {w*100:6.2f}% {c:9.3f}")
    tot_contrib += c
print(f"\nsum contrib: {tot_contrib:.3f}%")
