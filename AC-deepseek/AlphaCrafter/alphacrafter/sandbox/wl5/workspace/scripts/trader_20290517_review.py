from alphacrafter.sim.utils import get_account_dict
acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("market_value:", acc.get("market_value"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))
print("orders:", len(acc.get("orders", [])))
print("positions:")
for p in sorted(acc.get("positions", []), key=lambda x: x.get("market_value", 0), reverse=True):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} cost={p['cost_price']:.4f} cur={p['current_price']:.4f} mv={p['market_value']:.2f} pl={p['profit_loss']:.2f} plr={p['profit_loss_rate']*100:.2f}%")
