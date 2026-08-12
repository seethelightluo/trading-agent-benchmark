"""Trader block analysis: 2031-02-28 -> 2031-03-14 (10 td)."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

acc = get_account_dict()
total = acc["total_assets"]
positions = {p["symbol"]: p for p in acc["positions"]}

# Decision recon close = 2031-02-27 (previous completed day before 2031-02-28)
# Block end close = 2031-03-14
recon_date = "2031-02-27"
end_date = "2031-03-14"

rows = []
for sym, p in positions.items():
    df = get_stock_daily_data(symbol=sym, days=40)
    if df is None or len(df) == 0:
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    close_now = p["market_value"] / p["quantity"] if p["quantity"] else None
    # find close at recon and at end
    sub = df.loc[df.index <= pd.Timestamp(end_date)]
    if len(sub) == 0:
        continue
    close_end = float(sub["close"].iloc[-1])
    sub2 = df.loc[df.index <= pd.Timestamp(recon_date)]
    if len(sub2) == 0:
        continue
    close_recon = float(sub2["close"].iloc[-1])
    block_ret = close_end / close_recon - 1.0
    cur_w = p["market_value"] / total
    # estimated executed weight (back out block drift)
    exec_w = cur_w / (1.0 + block_ret) if abs(1.0 + block_ret) > 1e-9 else cur_w
    contrib = exec_w * block_ret
    rows.append({
        "sym": sym, "qty": p["quantity"], "cur_mv": p["market_value"],
        "cur_w": cur_w, "exec_w": exec_w, "block_ret": block_ret,
        "contrib": contrib, "pnl%": p.get("profit_loss_rate", 0),
    })

rows.sort(key=lambda r: r["contrib"], reverse=True)
print(f"total_assets: {total:.2f}  cash: {acc['available_cash']:.2f}")
print(f"{'sym':8s} {'exec_w%':>7s} {'cur_w%':>7s} {'block_ret%':>10s} {'contrib%':>9s}")
for r in rows:
    print(f"{r['sym']:8s} {r['exec_w']*100:7.2f} {r['cur_w']*100:7.2f} "
          f"{r['block_ret']*100:10.2f} {r['contrib']*100:9.2f}")
tot_c = sum(r["contrib"] for r in rows)
print(f"sum contrib: {tot_c*100:.2f}%")
