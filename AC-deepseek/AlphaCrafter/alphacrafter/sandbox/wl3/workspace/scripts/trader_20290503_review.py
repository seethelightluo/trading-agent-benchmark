from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets", acc.get("total_assets"))
print("net_assets", acc.get("net_assets"))
print("available_cash", acc.get("available_cash"))
print("gross_position_rate", acc.get("gross_position_rate"))
print("net_position_rate", acc.get("net_position_rate"))
print("n_positions", len(acc.get("positions", [])))
print("n_orders", len(acc.get("orders", [])))
pos = {p["symbol"]: p for p in acc.get("positions", [])}
for sym in sorted(pos.keys()):
    p = pos[sym]
    print(f"{sym}: qty={p['quantity']:.4f} mv={p['market_value']:.0f} pnl={p['profit_loss']:.0f} ({p['profit_loss_rate']*100:.2f}%)")
