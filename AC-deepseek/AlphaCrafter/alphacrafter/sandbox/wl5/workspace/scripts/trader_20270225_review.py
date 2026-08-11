"""Trader cycle review 2027-02-11 -> 2027-02-25: inspect account state."""
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("market_value:", acct.get("market_value"))
print("total_profit_loss:", acct.get("total_profit_loss"))
print("total_profit_loss_rate:", acct.get("total_profit_loss_rate"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("orders:", acct.get("orders"))
print("positions:")
for p in sorted(acct.get("positions", []), key=lambda x: x.get("market_value", 0), reverse=True):
    print(f"  {p['symbol']:10s} qty={p['quantity']:12.4f} px={p['current_price']:10.4f} "
          f"mv={p['market_value']:12.2f} pl={p['profit_loss']:12.2f} plr={p['profit_loss_rate']*100:7.2f}%")
