"""Trader account-state check at 2028-11-16 (read-only; no orders, no step)."""
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("market_value:", acct.get("market_value"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("watch_list:", acct.get("watch_list"))
print("positions:")
for p in acct.get("positions", []):
    print(f"  {p['symbol']:>10} qty={p['quantity']:>12.4f} mv={p['market_value']:>12.2f} "
          f"px={p['current_price']:>10.4f} pl={p['profit_loss']:>10.2f} ({p['profit_loss_rate']*100:.2f}%)")
print("orders:", acct.get("orders"))
