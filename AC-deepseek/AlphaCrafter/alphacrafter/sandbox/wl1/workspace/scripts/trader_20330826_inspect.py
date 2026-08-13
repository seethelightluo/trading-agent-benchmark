from alphacrafter.sim.utils import get_account_dict
import json

acct = get_account_dict()
print("date keys present:", [k for k in acct.keys()])
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("market_value:", acct.get("market_value"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("watch_list:", acct.get("watch_list"))
print("positions:")
for p in acct.get("positions", []):
    print("  ", p.get("symbol"), p.get("direction"), "qty", p.get("quantity"), "mv", round(p.get("market_value", 0), 2), "price", p.get("current_price"), "pnl%", p.get("profit_loss_rate"))
print("orders:")
for o in acct.get("orders", []):
    print("  ", o)
# try to find rebalance history / last rebalance marker
for k, v in acct.items():
    if k not in ("positions", "orders", "watch_list"):
        s = str(v)
        if len(s) < 400:
            print(k, ":", s)
