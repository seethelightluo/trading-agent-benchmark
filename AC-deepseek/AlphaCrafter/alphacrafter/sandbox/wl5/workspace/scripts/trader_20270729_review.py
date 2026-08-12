"""Trader review: 2027-07-29 -> 2027-08-12 block."""
import json
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("=== ACCOUNT ===")
print("total_assets:", round(acc.get("total_assets", 0), 2))
print("net_assets:", round(acc.get("net_assets", 0), 2))
print("available_cash:", round(acc.get("available_cash", 0), 2))
print("market_value:", round(acc.get("market_value", 0), 2))
print("total_profit_loss:", round(acc.get("total_profit_loss", 0), 2))
print("total_profit_loss_rate:", acc.get("total_profit_loss_rate"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))

print("\n=== POSITIONS ===")
pos = acc.get("positions", [])
tot_mv = sum(p.get("market_value", 0) for p in pos)
for p in sorted(pos, key=lambda x: -x.get("market_value", 0)):
    q = p.get("quantity", 0)
    if abs(q) < 1e-9:
        continue
    mv = p.get("market_value", 0)
    print(f"{p['symbol']:<10} qty={q:>12.4f} mv={mv:>12.2f} w={mv/tot_mv*100:>6.2f}% "
          f"pl={p.get('profit_loss',0):>12.2f} plr={p.get('profit_loss_rate',0)*100:>7.2f}%")

print("\n=== ORDERS ===")
orders = acc.get("orders", [])
print("pending orders:", len(orders))
for o in orders[:10]:
    print(o)

print("\n=== WATCHLIST ===")
print(acc.get("watch_list", []))
