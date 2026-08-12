from alphacrafter.sim.utils import get_account_dict
import json
acc = get_account_dict()
print('total_assets:', round(acc.get('total_assets'),2))
print('cash:', acc.get('available_cash'))
print('gross_position_rate:', acc.get('gross_position_rate'))
print('last_rebalance_date:', acc.get('last_rebalance_date'))
print('positions:')
for p in sorted(acc.get('positions', []), key=lambda x: -x['market_value']):
    print(' ', p['symbol'], round(p['quantity'],4), 'mv', round(p['market_value'],2), 'w%', round(100*p['market_value']/acc['net_assets'],2), 'pnl%', round(p.get('profit_loss_rate',0),2))
print('orders:', acc.get('orders'))
print('state:', open('trader_state.json').read())
