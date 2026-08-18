from alphacrafter.sim.utils import get_account_dict
acc = get_account_dict()
print("total_assets", round(acc.get("total_assets", 0), 2))
print("net_assets", round(acc.get("net_assets", 0), 2))
print("available_cash", round(acc.get("available_cash", 0), 4))
print("gross_position_rate", acc.get("gross_position_rate"))
print("net_position_rate", acc.get("net_position_rate"))
print("orders", len(acc.get("orders", [])))
pos = acc.get("positions", [])
pos.sort(key=lambda p: p.get("market_value", 0), reverse=True)
tot = sum(p.get("market_value", 0) for p in pos) or 1
for p in pos:
    mv = p.get("market_value", 0)
    print(f"{p['symbol']:10s} qty={p.get('quantity',0):12.4f} mv={mv:12.2f} w={mv/tot:6.3f} pl_rate={p.get('profit_loss_rate',0)*100:7.2f}%")
print("watch_list", acc.get("watch_list"))
