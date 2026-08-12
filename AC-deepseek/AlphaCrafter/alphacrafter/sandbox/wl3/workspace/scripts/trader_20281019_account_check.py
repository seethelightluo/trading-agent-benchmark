from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
print("total_assets", acct.get("total_assets"))
print("net_assets", acct.get("net_assets"))
print("available_cash", acct.get("available_cash"))
print("market_value", acct.get("market_value"))
print("gross_position_rate", acct.get("gross_position_rate"))
print("net_position_rate", acct.get("net_position_rate"))
print("total_profit_loss", acct.get("total_profit_loss"))
print("total_profit_loss_rate", acct.get("total_profit_loss_rate"))
print("orders", len(acct.get("orders", [])))
pos = acct.get("positions", [])
for p in sorted(pos, key=lambda x: -x.get("market_value", 0)):
    print(f"{p['symbol']}: qty={p['quantity']:.4f} mv={p['market_value']:.1f} pnl={p.get('profit_loss', 0):.1f} ({p.get('profit_loss_rate', 0) * 100:.2f}%)")
