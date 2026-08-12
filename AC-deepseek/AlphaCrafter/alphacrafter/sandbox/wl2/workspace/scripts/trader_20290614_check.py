from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))
print("orders:", acc.get("orders"))
pos = sorted(acc.get("positions", []), key=lambda p: -p.get("market_value", 0))
tot = sum(p.get("market_value", 0) for p in pos) or 1.0
for p in pos:
    print(f"{p['symbol']:>10s} qty={p['quantity']:>14.4f} cost={p['cost_price']:>10.4f} px={p['current_price']:>10.4f} mv={p['market_value']:>14.2f} wt={p['market_value']/tot:.4f} pnl%={p['profit_loss_rate']:8.2f}")
