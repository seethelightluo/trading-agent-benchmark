"""Inspect account state after step 2028-06-29 -> 2028-07-13."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acct = get_account_dict()
print("total_assets:", round(acct.get("total_assets", 0), 2))
print("net_assets:", round(acct.get("net_assets", 0), 2))
print("available_cash:", round(acct.get("available_cash", 0), 2))
print("market_value:", round(acct.get("market_value", 0), 2))
print("gross_position_rate:", round(acct.get("gross_position_rate", 0), 4))
print("net_position_rate:", round(acct.get("net_position_rate", 0), 4))
print("watch_list:", acct.get("watch_list", []))
print("orders:", acct.get("orders", []))

positions = {p["symbol"]: p for p in acct.get("positions", [])}
print("\npositions:")
tot_w = 0.0
for sym in acct.get("watch_list", []):
    p = positions.get(sym)
    if p is None:
        print(f"  {sym}: NO POSITION")
        continue
    qty = p.get("quantity", 0)
    mv = p.get("market_value", 0)
    w = mv / acct.get("total_assets", 1)
    tot_w += w
    print(f"  {sym}: qty={qty:.4f} cost={p.get('cost_price',0):.4f} px={p.get('current_price',0):.4f} "
          f"mv={mv:.2f} w={w:.4f} pl%={p.get('profit_loss_rate',0)*100:.2f}%")
print(f"\nsum weights: {tot_w:.4f}")
