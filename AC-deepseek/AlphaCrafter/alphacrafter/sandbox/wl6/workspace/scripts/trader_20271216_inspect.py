from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets", acc.get("total_assets"))
print("net_assets", acc.get("net_assets"))
print("available_cash", acc.get("available_cash"))
print("market_value", acc.get("market_value"))
print("gross_position_rate", acc.get("gross_position_rate"))
print("pending orders:", len(acc.get("orders", [])))
tot = 0.0
for p in sorted(acc.get("positions", []), key=lambda x: -x.get("market_value", 0)):
    q = p.get("quantity", 0)
    mv = p.get("market_value", 0)
    plr = p.get("profit_loss_rate", 0)
    print(f"{p['symbol']:8s} qty={q:12.4f} mv={mv:12.2f} plr={plr:8.2%}")
    tot += mv
print("sum mv", tot, "cash+mv", acc.get("available_cash", 0) + tot)
