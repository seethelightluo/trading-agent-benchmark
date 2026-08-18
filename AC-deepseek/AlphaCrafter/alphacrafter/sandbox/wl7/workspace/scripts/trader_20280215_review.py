from alphacrafter.sim.utils import get_account_dict
import json

acc = get_account_dict()
print("total_assets:", round(acc.get("total_assets", 0), 2))
print("net_assets:", round(acc.get("net_assets", 0), 2))
print("available_cash:", round(acc.get("available_cash", 0), 4))
print("gross_position_rate:", round(acc.get("gross_position_rate", 0), 4))
print("orders:", len(acc.get("orders", [])))
pos = acc.get("positions", [])
print("n_positions:", len(pos))
for p in sorted(pos, key=lambda x: -x.get("market_value", 0)):
    print(f"{p['symbol']:10s} qty={p.get('quantity',0):12.4f} mktval={p.get('market_value',0):12.2f} pl={p.get('profit_loss',0):10.2f} plr={p.get('profit_loss_rate',0)*100:7.2f}%")
