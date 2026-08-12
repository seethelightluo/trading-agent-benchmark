import json
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("net_assets:", acc.get("net_assets"))
print("total_assets:", acc.get("total_assets"))
print("available_cash:", acc.get("available_cash"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("watch_list:", acc.get("watch_list"))
print("num_positions:", len(acc.get("positions", [])))
for p in acc.get("positions", []):
    print(f"  {p['symbol']:8s} qty={p['quantity']:12.4f} mv={p['market_value']:12.2f} pl={p['profit_loss']:10.2f} ({p['profit_loss_rate']*100:+.2f}%)")
print("orders:", len(acc.get("orders", [])))
for o in acc.get("orders", [])[:5]:
    print("  ", o)
