from alphacrafter.sim.utils import get_account_dict
acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("market_value:", acc.get("market_value"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))
print("positions:")
for p in acc.get("positions", []):
    print(f"  {p['symbol']:10s} qty={p['quantity']:>12.4f} cost={p['cost_price']:>10.4f} px={p['current_price']:>10.4f} mktval={p['market_value']:>12.2f} pl={p['profit_loss']:>12.2f} plr={p['profit_loss_rate']:>8.4f}")
print("orders:", acc.get("orders"))
print("watch_list:", acc.get("watch_list"))
