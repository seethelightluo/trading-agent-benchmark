"""Screener 2027-05-28: compute strategy-consistent factor ICs, recent drift, correlations.

Uses the same factor definitions as strategy.py _asset_factor so IC evidence
matches what the trader will actually deploy. Read-only on data.
"""
import numpy as np
import pandas as pd
import pickle, json, math

p = pickle.load(open('scripts/panel_cache.pkl', 'rb'))
C, O, H, L, V = p['close'], p['open'], p['high'], p['low'], p['vol']
M = p['macro']
ret = C.pct_change()

SYMS = list(C.columns)
print('panel:', C.shape, C.index.min().date(), '->', C.index.max().date())
print('assets:', SYMS)

# ---------- strategy-consistent factor signals ----------
def nclv_nd(nd):
    return -(C - L.rolling(nd).min()) / (H.rolling(nd).max() - L.rolling(nd).min())

def rev_nd(nd):
    return -np.log(C / C.shift(nd))

signals = {
    'miner2_20260715_nclv_1d': nclv_nd(1),
    'miner2_20260715_nclv_2d': nclv_nd(2),
    'miner2_20260715_nclv_3d': nclv_nd(3),
    'miner2_20260715_nclv_5d': nclv_nd(5),
    'miner2_20260715_rev_1d': rev_nd(1),
    'miner2_20260715_rev_2d': rev_nd(2),
    'miner2_20260715_rev_3d': rev_nd(3),
    'miner2_20260715_rev_5d': rev_nd(5),
    'miner2_20260715_nbody_1d': -(C - O) / (H - L),
    'miner2_20260715_id_rev_1d': rev_nd(1),   # intraday reversal proxy (close-to-prev-close)
    'mom_120d_skip5': C.shift(5) / C.shift(125) - 1.0,
    'vol_of_vol20x60': ret.rolling(20).std().rolling(60).std(),
}
# vix beta conditional
vix = M['VIX']
vr = vix.pct_change()
beta = ret.rolling(60).cov(vr) / vr.rolling(60).var()
vm = vix / vix.shift(20) - 1.0
signals['vix_beta_cond_60x20'] = -beta * vm

for k, s in signals.items():
    signals[k] = s.replace([np.inf, -np.inf], np.nan)

# ---------- rank IC ----------
def rank_ic(sig, fwd, min_cov=8):
    """Daily cross-sectional Spearman IC between signal and forward return."""
    ics = {}
    for d in sig.index:
        s = sig.loc[d]
        f = fwd.loc[d]
        mask = s.notna() & f.notna()
        if mask.sum() < min_cov:
            continue
        ics[d] = s[mask].rank().corr(f[mask].rank())
    return pd.Series(ics)

def ic_table(fwd_label, fwd):
    rows = {}
    for name, sig in signals.items():
        ic = rank_ic(sig, fwd)
        if len(ic) < 30:
            rows[name] = None
            continue
        full = ic.mean()
        full_std = ic.std(ddof=1)
        icir = full / full_std if full_std > 0 else 0.0
        recent60 = ic.tail(60).mean()
        recent120 = ic.tail(120).mean()
        recent30 = ic.tail(30).mean()
        hit = (ic > 0).mean()
        rows[name] = dict(ic=full, icir=icir, n=len(ic),
                          ic30=recent30, ic60=recent60, ic120=recent120, hit=hit)
    return rows

fwd1 = ret.shift(-1)
fwd5 = ret.shift(-5)
fwd10 = ret.shift(-10)

t1 = ic_table('fwd1', fwd1)
t5 = ic_table('fwd5', fwd5)
t10 = ic_table('fwd10', fwd10)

print('\n=== Rank IC (strategy-consistent) ===')
print(f"{'factor':<28}{'ic1':>8}{'icir1':>8}{'ic5':>8}{'ic10':>8}{'ic1_30':>8}{'ic1_60':>8}{'ic1_120':>8}{'hit1':>7}")
for name in signals:
    if t1.get(name) is None:
        continue
    a, b, c = t1[name], t5[name], t10[name]
    print(f"{name:<28}{a['ic']:>8.4f}{a['icir']:>8.3f}{b['ic']:>8.4f}{c['ic']:>8.4f}"
          f"{a['ic30']:>8.4f}{a['ic60']:>8.4f}{a['ic120']:>8.4f}{a['hit']:>7.3f}")

# save for later
out = {'t1': t1, 't5': t5, 't10': t10}
with open('scripts/screener_20270528_ic_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nsaved scripts/screener_20270528_ic_results.json')

# ---------- regime assessment ----------
print('\n=== Regime Assessment (through 2027-05-27) ===')
for label, days in [('20d', 20), ('60d', 60), ('120d', 120)]:
    r = ret.tail(days)
    eq = [c for c in SYMS if c in ('000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX')]
    eqr = r[eq].mean(axis=1)
    print(f'{label}: mean ret {r.mean().mean()*100:+.2f}%/d | eq mean {eqr.mean()*100:+.2f}%/d | '
          f'ann vol eq {eqr.std()*np.sqrt(252)*100:.1f}% | last close vs 60d ago:')
    for s in SYMS:
        chg = (C[s].iloc[-1] / C[s].iloc[-1-60] - 1) * 100
        chg20 = (C[s].iloc[-1] / C[s].iloc[-1-20] - 1) * 100
        print(f'   {s:<10} 60d {chg:+6.1f}%  20d {chg20:+6.1f}%')

print('\nVIX last 5:', vix.dropna().tail(5).round(2).tolist())
print('VIX 20d chg %:', round((vix.iloc[-1]/vix.iloc[-21]-1)*100, 1))

# ---------- factor correlation (recent 120d, cross-sectional avg) ----------
print('\n=== Factor pairwise correlation (rank, avg across dates, last 120d) ===')
names = list(signals.keys())
corr_sum = {n: {} for n in names}
cnt = {n: {} for n in names}
for d in signals[names[0]].index[-120:]:
    vals = {}
    for n in names:
        s = signals[n].loc[d]
        vals[n] = s.rank()
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            v1, v2 = vals[n1], vals[n2]
            m = v1.notna() & v2.notna()
            if m.sum() < 8:
                continue
            c = v1[m].corr(v2[m])
            corr_sum[n1][n2] = corr_sum[n1].get(n2, 0) + c
            cnt[n1][n2] = cnt[n1].get(n2, 0) + 1
for n1 in names:
    for n2 in names:
        if n2 in corr_sum[n1] and cnt[n1][n2] > 0:
            corr_sum[n1][n2] /= cnt[n1][n2]
print('nclv_1d vs nclv_2d:', round(corr_sum['miner2_20260715_nclv_1d'].get('miner2_20260715_nclv_2d', float('nan')), 3))
print('nclv_1d vs nclv_3d:', round(corr_sum['miner2_20260715_nclv_1d'].get('miner2_20260715_nclv_3d', float('nan')), 3))
print('nclv_2d vs nclv_3d:', round(corr_sum['miner2_20260715_nclv_2d'].get('miner2_20260715_nclv_3d', float('nan')), 3))
print('rev_1d vs rev_2d:', round(corr_sum['miner2_20260715_rev_1d'].get('miner2_20260715_rev_2d', float('nan')), 3))
print('rev_2d vs rev_3d:', round(corr_sum['miner2_20260715_rev_2d'].get('miner2_20260715_rev_3d', float('nan')), 3))
print('nclv_1d vs rev_2d:', round(corr_sum['miner2_20260715_nclv_1d'].get('miner2_20260715_rev_2d', float('nan')), 3))
print('nclv_1d vs nbody_1d:', round(corr_sum['miner2_20260715_nclv_1d'].get('miner2_20260715_nbody_1d', float('nan')), 3))
print('mom_120d vs nclv_1d:', round(corr_sum['mom_120d_skip5'].get('miner2_20260715_nclv_1d', float('nan')), 3))
print('mom_120d vs vol_of_vol:', round(corr_sum['mom_120d_skip5'].get('vol_of_vol20x60', float('nan')), 3))
print('vol_of_vol vs vix_beta:', round(corr_sum['vol_of_vol20x60'].get('vix_beta_cond_60x20', float('nan')), 3))
print('mom_120d vs rev_2d:', round(corr_sum['mom_120d_skip5'].get('miner2_20260715_rev_2d', float('nan')), 3))
print('vix_beta vs rev_2d:', round(corr_sum['vix_beta_cond_60x20'].get('miner2_20260715_rev_2d', float('nan')), 3))
