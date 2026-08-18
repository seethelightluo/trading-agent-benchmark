"""Trader post-block attribution: account state after 08-17 -> 08-31 block."""
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets:", round(acc.get("total_assets", 0), 2))
print("net_assets:", round(acc.get("net_assets", 0), 2))
print("available_cash:", round(acc.get("available_cash", 0), 2))
print("market_value:", round(acc.get("market_value", 0), 2))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))
print("orders:", len(acc.get("orders", [])))
print("--- positions ---")
for p in sorted(acc.get("positions", []), key=lambda x: x.get("market_value", 0), reverse=True):
    print(f"{p['symbol']:10s} qty={p.get('quantity',0):>12.4f} mktval={p.get('market_value',0):>12.2f} "
          f"cost={p.get('cost_price',0):>10.4f} px={p.get('current_price',0):>10.4f} "
          f"pnl={p.get('profit_loss',0):>10.2f} plr={p.get('profit_loss_rate',0)*100:>7.2f}%")
