from alphacrafter.sim.utils import get_account_dict
acct = get_account_dict()
print('total_assets', round(acct.get('total_assets'),2))
print('cash', acct.get('available_cash'))
print('gross_position_rate', acct.get('gross_position_rate'))
print('pending orders:', len(acct.get('orders',[])))
print('positions:')
for p in acct.get('positions', []):
    print(' ', p['symbol'], round(p['quantity'],4), 'mv', round(p['market_value'],2), 'plr%', round(p.get('profit_loss_rate',0)*100,2))
# weight check
mv = sum(p['market_value'] for p in acct.get('positions',[]))
for p in acct.get('positions', []):
    print('  wt', p['symbol'], round(p['market_value']/mv*100,2))