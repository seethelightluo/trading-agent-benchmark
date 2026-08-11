"""Compute per-asset block PnL for 2027-05-20 -> 2027-06-03."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
acc = json.load(open("../persistent/account.json"))
pos = {p["symbol"]: p for p in acc["positions"]}

def get_df(sym):
    try:
        return get_stock_daily_data(sym, days=30) if sym not in OBS else get_index_daily_data(sym, days=30)
    except Exception:
        return None

rows = []
for sym, p in pos.items():
    df = get_df(sym)
    if df is None or len(df) < 2:
        rows.append((sym, 0.0, 0.0, 0.0, "no_data"))
        continue
    df = df.sort_values("date")
    c = df["close"].astype(float)
    d = df["date"].astype(str)
    idx_start = None
    for i, ds in enumerate(d):
        if ds >= "2027-05-20":
            idx_start = i
            break
    if idx_start is None:
        rows.append((sym, 0.0, 0.0, 0.0, "no_start"))
        continue
    p0 = float(c.iloc[idx_start])
    p1 = float(c.iloc[-1])
    qty = float(p["quantity"])
    pnl = qty * (p1 - p0)
    rows.append((sym, p0, p1, pnl, "ok"))

rows.sort(key=lambda r: -r[3])
tot = sum(r[3] for r in rows)
print(f"{'sym':10s} {'p0':>12s} {'p1':>12s} {'pnl':>12s}  status")
for sym, p0, p1, pnl, st in rows:
    print(f"{sym:10s} {p0:12.2f} {p1:12.2f} {pnl:12.2f}  {st}")
print(f"\nsum_block_pnl ~= {tot:.2f}")
print(f"net change: {1043719.1808 - 1037924.1538832588:.2f}")
