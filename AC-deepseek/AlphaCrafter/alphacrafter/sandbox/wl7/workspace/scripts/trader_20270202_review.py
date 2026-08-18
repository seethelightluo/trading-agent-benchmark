import json
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets", acct.get("total_assets"))
print("net_assets", acct.get("net_assets"))
print("cash", acct.get("available_cash"))
print("gross_pos", acct.get("gross_position_rate"))
print("n_positions", len(acct.get("positions", [])))
for p in sorted(acct.get("positions", []), key=lambda x: x.get("market_value", 0), reverse=True):
    print(f"{p['symbol']:12s} qty={p.get('quantity',0):12.4f} mv={p.get('market_value',0):12.2f} pnl={p.get('profit_loss',0):10.2f} ({p.get('profit_loss_rate',0)*100:6.2f}%)")
print("orders", len(acct.get("orders", [])))
