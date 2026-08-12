from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))
print("total_profit_loss:", acc.get("total_profit_loss"))
print("total_profit_loss_rate:", acc.get("total_profit_loss_rate"))
print("positions:")
for p in acc.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} cost={p['cost_price']:.4f} cur={p['current_price']:.4f} mv={p['market_value']:.2f} pnl={p['profit_loss']:.2f} pnl%={p['profit_loss_rate']*100:.2f}%")
print("orders:", acc.get("orders"))
print("watch_list:", acc.get("watch_list"))
