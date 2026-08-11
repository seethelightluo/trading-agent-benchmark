import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

acc = get_account_dict()
print("total_assets", round(acc["total_assets"], 2), "net", round(acc["net_assets"], 2))
print("cash", round(acc["available_cash"], 2), "gross_pos", round(acc["gross_position_rate"], 4))
print("positions:")
for p in acc["positions"]:
    print(" ", p["symbol"], "qty", round(p["quantity"], 4), "mv", round(p["market_value"], 2),
          "pnl", round(p.get("profit_loss", 0), 2), "pnl%", round(p.get("profit_loss_rate", 0) * 100, 2))
print("orders:", len(acc.get("orders", [])))
print("watch_list:", acc.get("watch_list", []))
