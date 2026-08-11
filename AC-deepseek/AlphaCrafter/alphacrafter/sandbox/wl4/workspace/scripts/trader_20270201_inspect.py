from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets:", round(acct.get("total_assets", 0), 2))
print("net_assets:", round(acct.get("net_assets", 0), 2))
print("available_cash:", round(acct.get("available_cash", 0), 2))
print("market_value:", round(acct.get("market_value", 0), 2))
print("gross_position_rate:", round(acct.get("gross_position_rate", 0), 4))
print("orders:", len(acct.get("orders", [])))
pos = acct.get("positions", [])
pos.sort(key=lambda p: p.get("market_value", 0), reverse=True)
tot = acct.get("net_assets", 0) or 1
for p in pos:
    print(f"  {p['symbol']:8s} qty={p['quantity']:>12.4f} px={p['current_price']:>12.4f} mv={p['market_value']:>14.2f} w={p['market_value']/tot:.4f}")
