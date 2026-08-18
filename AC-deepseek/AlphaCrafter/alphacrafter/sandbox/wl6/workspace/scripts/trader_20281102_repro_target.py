"""Reproduce the 2028-11-02 rebalance target from data visible through 2028-11-01.

Validates strategy.py logic against the executed last_executed_target_weights
in the live account (should match the 2028-11-02 executed target).
"""
import json
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, '.')
import strategy as S

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']


def load(sym):
    p = Path('../persistent/stock_data') / f'{sym}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def get_index(sym, days=300):
    df = load(sym)
    if df is None:
        return None
    return df.tail(days)


closes = {}
for a in WATCH:
    df = load(a)
    closes[a] = df['close'].astype(float) if df is not None else None

# visible through 2028-11-01 (decision on 2028-11-02)
cutoff = pd.Timestamp('2028-11-01')
closes_v = {a: (c[c.index <= cutoff] if c is not None else None) for a, c in closes.items()}

# replicate strategy computation
assets = WATCH
usable = [c.rename(a) for a, c in closes_v.items() if c is not None and len(c) >= 140]
panel = pd.concat(usable, axis=1, join='inner')

factors = S.load_ensemble()
print('ensemble factors:', [(f['factor_id'], f['weight'], f['direction']) for f in factors])

vix_df = get_index('VIX')
vix_close = vix_df['close'].astype(float) if vix_df is not None else None
raw = S.compute_raw_factors(closes_v, vix_close, assets)

score = {a: 0.0 for a in assets}
for f in factors:
    fid, w, d = f['factor_id'], f.get('weight', 0.0), f.get('direction', 1)
    r = S.rank_series(raw.get(fid, {}), assets)
    for a in assets:
        score[a] += (w * d) * (r[a] - 0.5)

regime = S.regime_from_market(panel)
print('regime at decision:', regime)
K = {'bull': 12, 'sideways': 10, 'bear': 8}[regime]
print('K:', K)

lo = min(score.values())
span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
print('top-K:', sorted(top))
print('score ranks:')
for a in sorted(assets, key=lambda a: score[a], reverse=True):
    print(f'  {a:10s} score={score[a]:+.4f} raw_w={raw_w[a]:.3f} in_top={a in top}')

w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
wsum = sum(w.values())
w = {a: v / wsum for a, v in w.items()}

vol20 = S.vol20_map(closes_v, assets)
valid_vol = {a: v for a, v in vol20.items() if v is not None and v > 0}
if valid_vol:
    vmin = min(valid_vol.values())
    inv = {a: (vmin / valid_vol[a] if a in valid_vol else 0.0) for a in assets}
    inv_top_sum = sum(inv.get(a, 0.0) for a in top)
    if inv_top_sum > 1e-12:
        blended = {a: ((1.0 - S.VOL_BLEND) * w.get(a, 0.0)
                       + S.VOL_BLEND * (inv.get(a, 0.0) / inv_top_sum if a in top else 0.0))
                   for a in assets}
        bsum = sum(blended.values())
        if bsum > 1e-12:
            w = {a: v / bsum for a, v in blended.items()}

frozen = S.frozen_set(closes_v, assets)
print('frozen:', frozen)
w = S.apply_floor(w, assets, [a for a in S.DEF if a in assets], S.FLOOR[regime])
w = S.apply_min_xau(w, assets)
for _ in range(50):
    prev = dict(w)
    w = S.apply_cap(w, assets)
    w = S.apply_crypto_cap(w, assets)
    w = S.apply_single_cap(w, assets, 'WTI', S.WTI_CAP)
    w = S.apply_frozen_cap(w, assets, frozen)
    w = S.apply_min_xau(w, assets)
    if sum(abs(w.get(a, 0.0) - prev.get(a, 0.0)) for a in assets) < 1e-11:
        break

total = sum(w.values())
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0 / len(assets))
           for a in assets}
rem = 1.0 - sum(weights.values())
weights[assets[0]] += rem

print('\nReproduced target:')
for a in assets:
    print(f'  {a:10s} {weights[a]*100:7.2f}%')

executed = json.load(open('../persistent/account.json'))['last_executed_target_weights']
print('\nExecuted (last_executed_target_weights):')
for a in assets:
    print(f'  {a:10s} {executed.get(a,0)*100:7.2f}%')

maxdiff = max(abs(weights[a] - executed.get(a, 0.0)) for a in assets)
print(f'\nmax |repro - executed| = {maxdiff:.5f}')
