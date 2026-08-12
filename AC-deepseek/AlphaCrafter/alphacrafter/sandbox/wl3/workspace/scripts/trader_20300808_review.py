from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("market_value:", acct.get("market_value"))
print("total_profit_loss:", acct.get("total_profit_loss"))
print("total_profit_loss_rate:", acct.get("total_profit_loss_rate"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("watch_list:", acct.get("watch_list"))
print("---POSITIONS---")
for p in sorted(acct.get("positions", []), key=lambda x: -abs(x.get("market_value", 0))):
    print(f"{p['symbol']:10s} qty={p.get('quantity',0):12.4f} cost={p.get('cost_price',0):10.4f} px={p.get('current_price',0):10.4f} mv={p.get('market_value',0):12.2f} pnl={p.get('profit_loss',0):10.2f} ({p.get('profit_loss_rate',0)*100:6.2f}%)")
print("---ORDERS---")
for o in acct.get("orders", []):
    print(o)
