import json
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("market_value:", acct.get("market_value"))
print("total_profit_loss:", acct.get("total_profit_loss"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("--- positions ---")
for p in sorted(acct.get("positions", []), key=lambda x: -x.get("market_value", 0)):
    print(f"{p['symbol']:10s} qty={p.get('quantity'):.3f} mv={p.get('market_value'):.2f} pnl={p.get('profit_loss'):.2f} ({p.get('profit_loss_rate')*100:.2f}%)")
print("--- orders ---")
for o in acct.get("orders", []):
    print(o)
