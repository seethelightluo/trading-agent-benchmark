import sys
sys.path.insert(0, '.')
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("gross_position_rate:", acct.get("gross_position_rate"))
pos = acct.get("positions", [])
print("n_positions:", len(pos))
for p in sorted(pos, key=lambda x: x.get("market_value", 0), reverse=True):
    print(" ", p.get("symbol"), "qty", round(p.get("quantity", 0), 4),
          "plr", round(p.get("profit_loss_rate", 0), 2),
          "mv", round(p.get("market_value", 0), 0))