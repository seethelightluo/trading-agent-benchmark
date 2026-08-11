"""Trader pre-step diagnostic for 2027-02-05 proposal day.

Checks: ensemble/strategy match, proposal eligibility, factor coverage,
regime, target weights (sum=1, non-negative, 15 assets).
"""
import json
import sys
sys.path.insert(0, '.')
import strategy as S

cur, tds = S._today_and_calendar()
print('current_date:', cur)
print('proposal eligible (should_propose):', S._should_propose(cur, tds))

ens = json.load(open('factor_ensemble.json'))['selected_factors']
print('ensemble factors:')
for x in ens:
    print(' ', x['factor_id'], x['weight'], 'dir', x['direction'])
print('strategy FACTORS:')
for fid, w, d in S.FACTORS:
    print(' ', fid, w, 'dir', d)

# verify match
ens_ids = {(x['factor_id'], x['weight'], x['direction']) for x in ens}
strat_ids = set(S.FACTORS)
print('ensemble == strategy:', ens_ids == strat_ids)

account = S.get_account_dict()
assets = list(account.get('watch_list', []))
print('n assets:', len(assets))

frames = S._fetch(assets)
scores, used = S._scores(frames, assets, cur)
print('factors used in composite:', used, '(need >=5)')
for a in assets:
    print('  %-10s score %.4f' % (a, scores[a]))

regime = S._regime(frames, assets)
print('regime:', regime)
w = S._weights(scores, assets, regime)
tot = sum(w.values())
print('weights: sum=%.8f min=%.6f max=%.6f n=%d' % (tot, min(w.values()), max(w.values()), len(w)))
for a in sorted(assets, key=lambda x: -w[x]):
    print('  %-10s %6.4f' % (a, w[a]))
f = S._forecasts(scores, assets)
print('forecast sample:', {a: round(f[a], 4) for a in assets[:5]})
print('diag OK')
