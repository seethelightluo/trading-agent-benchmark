from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("net_assets:", acct.get("net_assets"))
print("total_assets:", acct.get("total_assets"))
print("available_cash:", acct.get("available_cash"))
print("market_value:", acct.get("market_value"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("orders:", acct.get("orders"))
print("positions:")
for p in acct.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} mv={p['market_value']:.2f} cost={p['cost_price']:.4f} cur={p['current_price']:.4f}")
