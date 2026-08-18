"""Trader 2028-04-06: post-block account + performance attribution (read-only)."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("market_value:", acc.get("market_value"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("watch_list:", acc.get("watch_list"))
print("pending orders:", len(acc.get("orders", [])))
pos = acc.get("positions", [])
print("positions:")
for p in pos:
    print(f"  {p['symbol']}: qty={p.get('quantity'):.4f} mv={p.get('market_value'):.2f} "
          f"pnl={p.get('profit_loss'):.2f} ({p.get('profit_loss_rate')*100:.2f}%)")

# Block returns 2028-03-23 .. 2028-04-06 for attribution (visible up to today's close)
print("\nblock returns (2028-03-23 -> 2028-04-06):")
for a in acc.get("watch_list", []):
    f = get_stock_daily_data(a, days=300)
    if f is None or len(f) < 20:
        continue
    closes = f["close"].astype(float)
    # find 2028-03-23 and last row
    dates = f["date"]
    idx = None
    for i in range(len(dates) - 1, -1, -1):
        if str(dates.iloc[i])[:10] <= "2028-03-23":
            idx = i
            break
    if idx is None or idx >= len(closes) - 1:
        continue
    ret = closes.iloc[-1] / closes.iloc[idx] - 1.0
    print(f"  {a}: {ret*100:+.2f}%")
