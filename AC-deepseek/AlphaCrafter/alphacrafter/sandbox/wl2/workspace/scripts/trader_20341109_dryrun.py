"""Trader dry-run: compute the 2034-11-09 proposal (visible 11-08) without mutating state."""
import json
import sys
sys.path.insert(0, '.')
import strategy as st

date_state = json.load(open('../persistent/date.json'))
assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

# current weights from account
acc = json.load(open('../persistent/account.json'))
na = acc['net_assets']
cur = {}
for p in acc['positions']:
    cur[p['symbol']] = max(0.0, p['market_value']) / na

ens = st._load_ensemble()
print('ensemble factors:', [(e['factor_id'], round(e['weight'],4)) for e in ens])

built = st.build_target(assets, date_state, ens, current_weights=cur)
if built is None:
    print('build_target returned None')
    sys.exit(0)
weights, forecast, used, meta = built
print('used:', used)
print('risk %.3f vix %.1f m20 %.4f disp %.4f' % (meta['risk'], meta['vix'], meta['m20'], meta['disp']))
print('r20:', {a: round(v,4) for a, v in sorted(meta['r20'].items(), key=lambda x: -x[1])})
print('cap_map:', meta['cap_map'])
print('\ntarget weights (proposal):')
for a in sorted(weights, key=lambda x: -weights[x]):
    print('  %-10s %7.4f   cur %6.4f   delta %+6.4f   fc %+6.3f' % (a, weights[a], cur.get(a,0), weights[a]-cur.get(a,0), forecast[a]))
print('sum:', sum(weights.values()))
ow = sum(abs(weights[a]-cur.get(a,0)) for a in assets)
print('one-way turnover vs current: %.4f (%.1f%%)' % (ow, 100*ow))
edge = sum((weights[a]-cur.get(a,0))*forecast[a] for a in assets)
print('gross edge: %.2f bps vs threshold %.2f bps' % (10000*edge, 3*ow*10000))
