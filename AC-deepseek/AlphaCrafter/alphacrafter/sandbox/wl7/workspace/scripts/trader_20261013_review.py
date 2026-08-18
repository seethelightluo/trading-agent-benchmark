from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets", acc.get("total_assets"))
print("net_assets", acc.get("net_assets"))
print("available_cash", acc.get("available_cash"))
print("market_value", acc.get("market_value"))
print("gross_position_rate", acc.get("gross_position_rate"))
print("orders:", len(acc.get("orders", [])))
for o in acc.get("orders", [])[:6]:
    print("  order:", o.get("symbol"), o.get("order_type"), o.get("quantity"), o.get("status"))
print("positions:")
for p in acc.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} mv={p['market_value']:.0f} pnl={p['profit_loss']:.0f} ({p['profit_loss_rate']*100:.2f}%)")
