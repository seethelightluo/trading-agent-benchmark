"""Trader probe: current account state and date (read-only)."""
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("watch_list:", acct.get("watch_list"))
print("positions:")
for p in acct.get("positions", []):
    print("  ", p.get("symbol"), p.get("direction"), "qty=", p.get("quantity"),
          "mv=", round(p.get("market_value", 0), 2),
          "cost=", round(p.get("cost_price", 0), 4),
          "px=", round(p.get("current_price", 0), 4),
          "pnl=", round(p.get("profit_loss", 0), 2))
print("orders:", acct.get("orders"))
