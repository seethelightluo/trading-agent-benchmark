from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
pos = {p["symbol"]: p for p in acct.get("positions", [])}
for a in acct.get("watch_list", []):
    p = pos.get(a)
    if p:
        print(f"{a:8s} qty {p['quantity']:14.4f}  cost {p['cost_price']:.4f}  plr {p['profit_loss_rate']*100:7.2f}%  pl {p['profit_loss']:12,.2f}")
