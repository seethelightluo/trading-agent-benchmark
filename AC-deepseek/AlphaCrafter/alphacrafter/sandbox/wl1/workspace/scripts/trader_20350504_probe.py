"""Probe current account + calendar state for the trader (2035-05-04)."""
import json
from alphacrafter.sim.utils import get_account_dict

d = json.load(open("../persistent/date.json"))
cur = str(d["current_date"])
tds = d.get("trading_days", [])

print("current_date:", cur)
if cur in tds:
    i = tds.index(cur)
    print("idx in trading_days:", i)
    for k in range(max(0, i - 12), min(len(tds), i + 3)):
        print("   ", tds[k])
else:
    print("WARNING: current date not in trading_days list")

acc = get_account_dict()
print("\n--- account ---")
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("market_value:", acc.get("market_value"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("last_rebalance_date:", acc.get("last_rebalance_date"))
print("n_positions:", len(acc.get("positions", [])))
for p in acc.get("positions", []):
    print(f"  {p['symbol']:10s} qty={p['quantity']:12.4f} mv={p['market_value']:12.2f} pnl={p['profit_loss']:10.2f}")
print("n_orders:", len(acc.get("orders", [])))
print("watch_list:", acc.get("watch_list"))
