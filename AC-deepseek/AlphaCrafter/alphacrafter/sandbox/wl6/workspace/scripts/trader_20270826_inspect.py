from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("net_assets:", acct.get("net_assets"))
print("total_assets:", acct.get("total_assets"))
print("available_cash:", acct.get("available_cash"))
print("market_value:", acct.get("market_value"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("total_profit_loss:", acct.get("total_profit_loss"))
print("orders:", acct.get("orders"))
print("--- positions ---")
for p in sorted(acct.get("positions", []), key=lambda x: -abs(x.get("market_value", 0))):
    print(f"{p['symbol']:10s} qty={p.get('quantity'):10.4f} cost={p.get('cost_price'):10.4f} px={p.get('current_price'):10.4f} mv={p.get('market_value'):12.2f} pnl={p.get('profit_loss'):12.2f} rate={p.get('profit_loss_rate'):8.2f}%")
