from alphacrafter.sim.utils import get_account_dict
acc = get_account_dict()
print('total_assets:', acc.get('total_assets'))
print('net_assets:', acc.get('net_assets'))
print('available_cash:', acc.get('available_cash'))
print('market_value:', acc.get('market_value'))
print('gross_position_rate:', acc.get('gross_position_rate'))
print('last_rebalance_date:', acc.get('last_rebalance_date'))
print('watch_list:', acc.get('watch_list'))
print('positions:')
for p in acc.get('positions', []):
    print(' ', p['symbol'], p['direction'], round(p['quantity'],4), 'mv', round(p['market_value'],2), 'w%', round(100*p['market_value']/acc['net_assets'],2) if acc['net_assets'] else 0)
print('orders:', acc.get('orders'))
