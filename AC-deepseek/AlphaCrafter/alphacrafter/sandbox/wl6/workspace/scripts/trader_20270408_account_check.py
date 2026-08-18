"""Trader post-step account inspection 2027-04-08."""
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("pending orders:", len(acct.get("orders", [])))
print("--- positions ---")
for p in sorted(acct.get("positions", []), key=lambda x: -abs(x.get("market_value", 0))):
    print(f"{p['symbol']:>10s} qty={p.get('quantity',0):>12.4f} "
          f"mktval={p.get('market_value',0):>12.2f} "
          f"pnl_rate={p.get('profit_loss_rate',0)*100:>8.2f}% "
          f"pnl={p.get('profit_loss',0):>10.2f}")
