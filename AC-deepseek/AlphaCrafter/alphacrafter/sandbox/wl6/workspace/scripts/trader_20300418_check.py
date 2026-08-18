from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets:", round(acct["total_assets"], 2))
print("net_assets:", round(acct["net_assets"], 2))
print("available_cash:", acct["available_cash"])
print("positions:")
for p in acct.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} mv={p['market_value']:.0f} w={p['market_value']/acct['net_assets']*100:.2f}% pl={p.get('profit_loss',0):.0f} plr={p.get('profit_loss_rate',0)*100:.2f}%")
print("orders:", len(acct.get("orders", [])))
