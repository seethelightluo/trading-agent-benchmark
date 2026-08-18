from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("market_value:", acc.get("market_value"))
print("total_profit_loss:", acc.get("total_profit_loss"))
print("total_profit_loss_rate:", acc.get("total_profit_loss_rate"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))
print("watch_list:", acc.get("watch_list"))
print("---POSITIONS---")
for p in acc.get("positions", []):
    print(p.get("symbol"), "qty=", round(p.get("quantity",0),4),
          "cost=", p.get("cost_price"), "px=", p.get("current_price"),
          "mv=", round(p.get("market_value",0),2),
          "plr=", round(p.get("profit_loss_rate",0)*100,2))
print("---ORDERS---")
for o in acc.get("orders", []):
    print(o)
