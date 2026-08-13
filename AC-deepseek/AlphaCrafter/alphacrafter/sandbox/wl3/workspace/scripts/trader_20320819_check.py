from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets", acc.get("total_assets"))
print("net_assets", acc.get("net_assets"))
print("available_cash", acc.get("available_cash"))
print("market_value", acc.get("market_value"))
print("gross_position_rate", acc.get("gross_position_rate"))
print("net_position_rate", acc.get("net_position_rate"))
print("total_profit_loss", acc.get("total_profit_loss"))
print("total_profit_loss_rate", acc.get("total_profit_loss_rate"))
print("n_positions", len(acc.get("positions", [])))
for p in sorted(acc.get("positions", []), key=lambda x: -abs(x.get("market_value", 0))):
    print(
        f"  {p['symbol']}: qty={p.get('quantity'):.4f} mv={p.get('market_value', 0):,.0f} "
        f"pnl={p.get('profit_loss', 0):,.0f} ({p.get('profit_loss_rate', 0) * 100:.2f}%)"
    )
print("n_orders", len(acc.get("orders", [])))
for o in acc.get("orders", [])[:5]:
    print("  order:", o)
