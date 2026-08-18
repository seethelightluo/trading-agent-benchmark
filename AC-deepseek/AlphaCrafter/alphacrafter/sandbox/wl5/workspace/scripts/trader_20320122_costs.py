from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
positions = {p["symbol"]: p for p in acc.get("positions", [])}
print("net_assets", acc.get("net_assets"))
for s in ["ETH", "BTC", "SOX", "NDX", "000300.SH", "US10Y"]:
    p = positions.get(s)
    if p:
        print(f"{s}: qty={p['quantity']:.6f} cost={p['cost_price']:.4f} cur={p['current_price']:.4f} plr={p['profit_loss_rate']*100:.2f}%")
