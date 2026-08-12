from alphacrafter.sim.utils import get_account_dict
acc = get_account_dict()
print("net_assets:", acc.get("net_assets"))
print("total_assets:", acc.get("total_assets"))
print("available_cash:", acc.get("available_cash"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("total_profit_loss:", acc.get("total_profit_loss"))
print("total_profit_loss_rate:", acc.get("total_profit_loss_rate"))
print("positions:")
tot = 0
for p in acc.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} mv={p['market_value']:.0f} pl={p['profit_loss']:.0f} ({p['profit_loss_rate']*100:.2f}%)")
    tot += p['profit_loss']
print("sum pos pl:", tot)
print("orders:", acc.get("orders"))
print("watch_list:", acc.get("watch_list"))
