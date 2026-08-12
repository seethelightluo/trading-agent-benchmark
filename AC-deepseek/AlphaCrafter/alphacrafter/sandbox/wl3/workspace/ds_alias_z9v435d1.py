
from alphacrafter.sim.utils import get_account_dict
import json
acc = get_account_dict()
print("net_assets:", acc.get("net_assets"))
print("total_assets:", acc.get("total_assets"))
print("available_cash:", acc.get("available_cash"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))
print("total_profit_loss:", acc.get("total_profit_loss"))
print("total_profit_loss_rate:", acc.get("total_profit_loss_rate"))
print("n_positions:", len(acc.get("positions", [])))
print("n_orders:", len(acc.get("orders", [])))
pos = sorted(acc.get("positions", []), key=lambda p: p.get("market_value", 0), reverse=True)
for p in pos[:20]:
    print(p["symbol"], "qty=", round(p.get("quantity",0),4), "mv=", round(p.get("market_value",0),1), "pnl=", round(p.get("profit_loss",0),1), "pnl%=", round(p.get("profit_loss_rate",0)*100,2))
