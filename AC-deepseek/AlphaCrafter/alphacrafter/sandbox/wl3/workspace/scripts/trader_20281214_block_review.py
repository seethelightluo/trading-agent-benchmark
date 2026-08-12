import json
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
nav = acct.get("net_assets", 0)
print("net_assets:", round(nav, 2))
print("total_assets:", round(acct.get("total_assets", 0), 2))
print("available_cash:", round(acct.get("available_cash", 0), 2))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("total_pnl:", round(acct.get("total_profit_loss", 0), 2))
print("total_pnl_rate:", acct.get("total_profit_loss_rate"))
print("orders:", len(acct.get("orders", [])))
print("positions:")
tot = 0
for p in sorted(acct.get("positions", []), key=lambda x: -x.get("market_value", 0)):
    q = p.get("quantity", 0)
    mv = p.get("market_value", 0)
    pl = p.get("profit_loss", 0)
    plr = p.get("profit_loss_rate", 0)
    tot += mv
    print(f"  {p['symbol']:>10s} qty={q:>12.4f} mv={mv:>12.2f} pl={pl:>12.2f} plr={plr*100:>7.2f}% w={mv/nav*100:.2f}%")
print("sum mv:", round(tot, 2), "cash+mv:", round(tot + acct.get("available_cash", 0), 2))
