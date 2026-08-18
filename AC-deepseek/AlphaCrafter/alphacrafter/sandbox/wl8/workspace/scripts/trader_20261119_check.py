"""Post-step account/position check as of 2026-11-19."""
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets", round(acc.get("total_assets", 0), 2))
print("net_assets", round(acc.get("net_assets", 0), 2))
print("available_cash", round(acc.get("available_cash", 0), 4))
print("gross_position_rate", acc.get("gross_position_rate"))
print("positions:")
tot_mv = 0.0
for p in acc.get("positions", []):
    tot_mv += p.get("market_value", 0)
    print(" ", p["symbol"], p["direction"], round(p.get("quantity", 0), 4),
          "px", round(p.get("current_price", 0), 4),
          "mv", round(p.get("market_value", 0), 2),
          "w", round(p.get("market_value", 0) / max(1, acc.get("net_assets", 1)), 4),
          "pnl%", round(p.get("profit_loss_rate", 0) * 100, 2))
print("sum mv", round(tot_mv, 2))
print("orders:", acc.get("orders", []))