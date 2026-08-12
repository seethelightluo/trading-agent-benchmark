from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("market_value:", acc.get("market_value"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("positions:")
for p in sorted(acc.get("positions", []), key=lambda x: -x.get("market_value", 0)):
    print(f"  {p['symbol']:10s} qty={p.get('quantity',0):12.4f} mv={p.get('market_value',0):12.2f} pl={p.get('profit_loss',0):10.2f} pl%={p.get('profit_loss_rate',0)*100:7.2f}% cost={p.get('cost_price',0):10.4f} px={p.get('current_price',0):10.4f}")
print("orders:", acc.get("orders"))
print("watch_list:", acc.get("watch_list"))
