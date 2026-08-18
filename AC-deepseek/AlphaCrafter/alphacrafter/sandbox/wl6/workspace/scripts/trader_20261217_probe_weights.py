"""Trader probe: verify updated strategy computes valid 15-asset target weights."""
import sys
sys.path.insert(0, '.')
import importlib.util
from math import isfinite
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

spec = importlib.util.spec_from_file_location('strat', 'strategy.py')
# We won't execute the hook; just import helpers by parsing? Simpler: replicate quickly.
import strategy as S

WL = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames = {a: S.stock(a) for a in WL}
closes = {a: (f.close.astype(float) if f is not None and 'close' in f else None) for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
panel = pd.concat(usable, axis=1, join='inner')
factors = S.load_ensemble()
print('ensemble:', [(f['factor_id'], f.get('weight'), f.get('direction')) for f in factors])
vf = S.index('VIX')
vix_close = vf.close.astype(float) if vf is not None and 'close' in vf else None
raw = S.compute_raw_factors(closes, vix_close, WL)
score = {a: 0.0 for a in WL}
for f in factors:
    fid, w, d = f['factor_id'], f.get('weight', 0.0), f.get('direction', 1)
    r = S.rank_series(raw.get(fid, {}), WL)
    for a in WL:
        score[a] += (w * d) * (r[a] - 0.5)

regime = S.regime_from_market(panel)
K = {'bull': 12, 'sideways': 10, 'bear': 8}[regime]
print('regime:', regime, 'K:', K)
lo = min(score.values()); span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in WL}
top = set(sorted(WL, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
w = {a: (raw_w[a] if a in top else 0.0) for a in WL}
if sum(w.values()) < 1e-9:
    w = {a: (1.0 / K if a in top else 0.0) for a in WL}
w = S.apply_floor(w, WL, [a for a in S.DEF if a in WL], S.FLOOR[regime])
w = S.apply_cap(w, WL)
total = sum(w.values())
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0/len(WL)) for a in WL}
rem = 1.0 - sum(weights.values()); weights[WL[0]] += rem

print('\nProposed target weights (sum=%.6f):' % sum(weights.values()))
for a in WL:
    print(f'  {a:10s} {weights[a]*100:6.2f}%')
print('\nDefensive (XAU/US10Y/CN10Y) total: %.2f%%' % (sum(weights[a] for a in S.DEF)*100))
print('Max single weight: %.2f%%' % (max(weights.values())*100))
print('Nonzero count:', sum(1 for v in weights.values() if v > 1e-6))
