from alphacrafter.sim.utils import get_account_dict
acct = get_account_dict()
print('Net assets:', acct.get('net_assets'))
print('Available cash:', acct.get('available_cash'))
print('Gross position rate:', acct.get('gross_position_rate'))
total = acct.get('net_assets')
for p in sorted(acct.get('positions', []), key=lambda x: x.get('market_value',0), reverse=True):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f}, mkt_val={p.get('market_value',0):.1f} ({p.get('market_value',0)/total*100:.1f}%), pnl={p.get('profit_loss',0):+.1f}")