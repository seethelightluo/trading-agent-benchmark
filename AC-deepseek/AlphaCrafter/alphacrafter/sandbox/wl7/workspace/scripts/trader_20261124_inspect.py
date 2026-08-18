from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("total_profit_loss:", acc.get("total_profit_loss"))
print("total_profit_loss_rate:", acc.get("total_profit_loss_rate"))
print("--- POSITIONS ---")
for p in sorted(acc.get("positions", []), key=lambda x: -x.get("market_value", 0)):
    print(f"{p['symbol']:10s} qty={p.get('quantity',0):12.4f} mv={p.get('market_value',0):14.2f} pnl={p.get('profit_loss',0):12.2f} ({p.get('profit_loss_rate',0)*100:7.2f}%)")
print("--- ORDERS ---")
for o in acc.get("orders", []):
    print(o)
print("watch_list:", acc.get("watch_list"))
