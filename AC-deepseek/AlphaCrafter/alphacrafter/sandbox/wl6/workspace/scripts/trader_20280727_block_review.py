from alphacrafter.sim.utils import get_account_dict
import json

acct = get_account_dict()
print("total_assets", round(acct.get("total_assets", 0), 2))
print("available_cash", round(acct.get("available_cash", 0), 2))
print("market_value", round(acct.get("market_value", 0), 2))
print("pnl", round(acct.get("total_profit_loss", 0), 2))
print("gross_position_rate", acct.get("gross_position_rate"))
positions = {p["symbol"]: p for p in acct.get("positions", [])}
print("\nPOSITIONS (quantity, cost, cur, mktval, pnl%, w%):")
tot = acct.get("total_assets", 1)
for s, p in sorted(positions.items(), key=lambda x: -x[1]["market_value"]):
    print(f"{s:10s} qty={p['quantity']:10.2f} cost={p['cost_price']:9.3f} "
          f"cur={p['current_price']:9.3f} mv={p['market_value']:10.0f} "
          f"pnl%={p['profit_loss_rate']:7.2f} w={100*p['market_value']/tot:5.1f}%")
print("\nORDERS:", json.dumps(acct.get("orders", []))[:800])
