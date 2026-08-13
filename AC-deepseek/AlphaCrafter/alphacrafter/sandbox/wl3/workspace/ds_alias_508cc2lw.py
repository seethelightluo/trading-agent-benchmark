
from alphacrafter.sim.utils import get_account_dict
import json
acct = get_account_dict()
print("net_assets:", acct.get("net_assets"))
print("total_assets:", acct.get("total_assets"))
print("available_cash:", acct.get("available_cash"))
print("market_value:", acct.get("market_value"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("total_profit_loss:", acct.get("total_profit_loss"))
print("total_profit_loss_rate:", acct.get("total_profit_loss_rate"))
print("positions:")
for p in acct.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} mv={p['market_value']:.2f} pnl={p.get('profit_loss',0):.2f} ({p.get('profit_loss_rate',0)*100:.2f}%)")
print("orders:", len(acct.get("orders", [])))
print("watch_list:", acct.get("watch_list"))
