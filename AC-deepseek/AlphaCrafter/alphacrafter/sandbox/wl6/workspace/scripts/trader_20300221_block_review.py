from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("orders:", len(acc.get("orders", [])))
for p in sorted(acc.get("positions", []), key=lambda x: -x.get("market_value", 0)):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} mv={p['market_value']:.0f} pnl={p.get('profit_loss',0):.0f} pnl%={p.get('profit_loss_rate',0)*100:.2f}")
