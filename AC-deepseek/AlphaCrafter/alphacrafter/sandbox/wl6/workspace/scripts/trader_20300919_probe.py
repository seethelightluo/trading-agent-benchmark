"""Trader probe: inspect account state after 2030-09-05 -> 2030-09-19 step."""
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets", acct.get("total_assets"))
print("net_assets", acct.get("net_assets"))
print("available_cash", acct.get("available_cash"))
print("market_value", acct.get("market_value"))
print("gross_position_rate", acct.get("gross_position_rate"))
print("net_position_rate", acct.get("net_position_rate"))
print("total_profit_loss", acct.get("total_profit_loss"))
print("total_profit_loss_rate", acct.get("total_profit_loss_rate"))
print("watch_list", acct.get("watch_list"))
print("--- positions ---")
for p in acct.get("positions", []):
    print(p.get("symbol"), p.get("direction"), round(p.get("quantity", 0), 4),
          "px", round(p.get("current_price", 0), 4),
          "mv", round(p.get("market_value", 0), 2),
          "pnl", round(p.get("profit_loss", 0), 2),
          "pnl%", round(p.get("profit_loss_rate", 0) * 100, 2))
print("--- orders ---")
for o in acct.get("orders", []):
    print(o)
