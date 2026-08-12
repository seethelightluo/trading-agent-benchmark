from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))
print("watch_list:", acc.get("watch_list"))
print("orders:", acc.get("orders"))
print("--- positions ---")
for p in acc.get("positions", []):
    print(f"{p['symbol']:10s} qty={p['quantity']:>14.6f} cost={p['cost_price']:>12.4f} cur={p['current_price']:>12.4f} mv={p['market_value']:>14.2f} pl_pct={p['profit_loss_rate']*100:>8.2f}")
