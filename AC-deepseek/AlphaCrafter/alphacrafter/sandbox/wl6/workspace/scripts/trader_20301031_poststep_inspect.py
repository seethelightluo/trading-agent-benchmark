"""Post-step account inspection: determine if a rebalance executed on 2030-10-17."""
from alphacrafter.sim.utils import get_account_dict
import pandas as pd

acct = get_account_dict()
assets = acct.get("watch_list", [])
print("account:", {k: acct.get(k) for k in ("total_assets","net_assets","available_cash","gross_position_rate","net_position_rate")})
print("pending orders:", len(acct.get("orders",[])))
for p in acct.get("orders",[]):
    print("  order:", p)
positions = {p["symbol"]: p for p in acct.get("positions",[])}
for a in assets:
    p = positions.get(a)
    if p:
        print(f"{a}: qty={p['quantity']:.3f} mv={p['market_value']:.0f} pnl={p['profit_loss']:.0f} ({p['profit_loss_rate']*100:.2f}%)")
    else:
        print(f"{a}: NO POSITION")

# weight view
mv = sum(positions[a]["market_value"] for a in assets if a in positions)
print("\ntotal mv:", round(mv,0), "implied weights:")
if mv > 0:
    for a in assets:
        if a in positions:
            print(f"  {a}: {positions[a]['market_value']/mv*100:.2f}%")