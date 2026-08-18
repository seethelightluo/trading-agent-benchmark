from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
for p in sorted(acc.get("positions", []), key=lambda x: -x.get("market_value", 0)):
    q = p.get("quantity", 0)
    if abs(q) < 1e-9:
        continue
    print(f"{p['symbol']}: qty={q:.4f} cost={p.get('cost_price',0):.4f} cur={p.get('current_price',0):.4f} pnl%={p.get('profit_loss_rate',0)*100:.2f}")
