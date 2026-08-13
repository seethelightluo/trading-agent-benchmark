"""miner1 2032-04-09: broad screen of candidate factor families on 15-name panel.
Gate: abs daily rank IC >= 0.0070 and abs ICIR >= 0.0840 (full-sample paper metrics)."""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr

panel = pd.read_pickle('scripts/panel_cache_20320409.pkl')
close = panel['close']; high = panel['high']; low = panel['low']; opn = panel['open']
vol = panel['vol']; ret = panel['ret']; macro = panel['macro']

def fwd_ret(h):
    return close.pct_change(h).shift(-h)

def rank_ic(factor, fwd, min_n=8):
    ics, dates = [], []
    common = factor.index.intersection(fwd.index)
    for d in common:
        f = factor.loc[d].astype(float)
        r = fwd.loc[d].astype(float)
        m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if m.sum() >= min_n and f[m].nunique() > 1 and r[m].nunique() > 1:
            ic = spearmanr(f[m], r[m]).correlation
            if np.isfinite(ic):
                ics.append(ic); dates.append(d)
    ics = np.array(ics)
    if len(ics) < 30:
        return None
    sd = ics.std(ddof=1)
    return {'ic': ics.mean(), 'icir': ics.mean()/sd if sd > 0 else np.nan,
            'hit': (np.sign(ics) == np.sign(ics.mean())).mean(), 'n': len(ics)}

def coverage(factor):
    valid = factor.notna() & np.isfinite(factor)
    return valid.mean().mean()

def evaluate(name, factor, horizons=(1, 5, 10)):
    out = {'name': name}
    for h in horizons:
        r = rank_ic(factor, fwd_ret(h))
        if r:
            out[f'ic{h}'] = r['ic']; out[f'icir{h}'] = r['icir']
            out[f'hit{h}'] = r['hit']; out[f'n{h}'] = r['n']
    out['cov'] = coverage(factor)
    r = rank_ic(factor.loc['2030-04-01':], fwd_ret(1).loc['2030-04-01':])
    if r:
        out['ic1_recent2y'] = r['ic']; out['icir1_recent2y'] = r['icir']
    return out

def mk(name, f):
    res = evaluate(name, f)
    if res.get('ic1') is not None:
        print(f"{name:34s} ic1={res.get('ic1', float('nan')):+.4f} icir1={res.get('icir1', float('nan')):+.3f} "
              f"hit1={res.get('hit1', float('nan')):.3f} n1={res.get('n1', 0)} "
              f"ic5={res.get('ic5', float('nan')):+.4f} ic10={res.get('ic10', float('nan')):+.4f} "
              f"cov={res.get('cov', float('nan')):.2f} | recent2y ic1={res.get('ic1_recent2y', float('nan')):+.4f} "
              f"icir={res.get('icir1_recent2y', float('nan')):+.3f}")
    return res

print("=== Candidate scan (full sample 2020-01-01..2032-04-08) ===")
results = {}

for n in (10, 20, 60, 120):
    results[f'mom_{n}d'] = mk(f'mom_{n}d', close / close.shift(n) - 1)
results['mom_120d_skip5'] = mk('mom_120d_skip5', close / close.shift(125) - 1)

for n in (20, 60):
    voln = ret.rolling(n).std()
    results[f'rs_mom_{n}'] = mk(f'rs_mom_{n}', (close / close.shift(n) - 1) / voln)

for n in (10, 20, 60):
    ma = close.rolling(n).mean()
    results[f'ma_rev_{n}'] = mk(f'ma_rev_{n}', -(close / ma - 1))

for n in (20, 60):
    results[f'zscore_{n}'] = mk(f'zscore_{n}', (close - close.rolling(n).mean()) / (ret.rolling(n).std()))

body = (close - opn) / opn
results['body_rev_1d'] = mk('body_rev_1d', -body)
rng = (high - low).replace(0, np.nan)
upper_wick = (high - np.maximum(opn, close)) / rng
results['upper_wick_1d'] = mk('upper_wick_1d', upper_wick)
lower_wick = (np.minimum(opn, close) - low) / rng
results['lower_wick_1d'] = mk('lower_wick_1d', lower_wick)

volma = vol.rolling(20).mean()
results['vol_z_20'] = mk('vol_z_20', vol / volma - 1)

def trend_persist(n5=5, n20=20, n60=60):
    s5 = np.sign(close / close.shift(n5) - 1)
    s20 = np.sign(close / close.shift(n20) - 1)
    s60 = np.sign(close / close.shift(n60) - 1)
    return (s5 + s20 + s60) / 3.0
results['trend_persist_5_20_60'] = mk('trend_persist_5_20_60', trend_persist())

hv = ret.rolling(20).std()
hv_med = hv.median(axis=1)
rev1 = -ret
results['rev1_x_highvol'] = mk('rev1_x_highvol', rev1.where(hv > hv_med, 0.0))
results['rev1_x_volz'] = mk('rev1_x_volz', -ret * (vol / volma - 1).clip(-3, 3))

vix_chg = macro['VIX'].pct_change(5)
vix_cond = vix_chg.reindex(ret.index).ffill()
results['rev1_x_vixup'] = mk('rev1_x_vixup', rev1.where(vix_cond > 0, 0.0))

dxy_ret = macro['DXY'].pct_change()
dxy_ret = dxy_ret.reindex(ret.index).ffill()
beta_dxy = ret.rolling(60).cov(dxy_ret) / dxy_ret.rolling(60).var()
results['dxy_beta_60'] = mk('dxy_beta_60', beta_dxy)
results['dxy_beta_x_dxymom'] = mk('dxy_beta_x_dxymom', beta_dxy * dxy_ret.rolling(20).mean())

us10y_ret = ret['US10Y']
beta_10y = ret.rolling(60).cov(us10y_ret) / us10y_ret.rolling(60).var()
results['us10y_beta_60'] = mk('us10y_beta_60', beta_10y)
results['us10y_beta_x_10ymom'] = mk('us10y_beta_x_10ymom', beta_10y * us10y_ret.rolling(20).mean())

for n in (60, 120):
    dd = close / close.rolling(n).max() - 1
    results[f'dd_{n}'] = mk(f'dd_{n}', dd)

results['rev_1d'] = mk('rev_1d', -ret)
results['rev_2d'] = mk('rev_2d', -(close / close.shift(2) - 1))

print("\n=== Passed gate (|ic1|>=0.007 & |icir1|>=0.084) ===")
for k, v in results.items():
    ic1 = v.get('ic1', 0); icir1 = v.get('icir1', 0)
    if abs(ic1) >= 0.007 and abs(icir1) >= 0.084:
        print(f"  {k}: ic1={ic1:+.4f} icir1={icir1:+.3f} hit1={v.get('hit1', 0):.3f} n1={v.get('n1', 0)} cov={v.get('cov', 0):.2f}")
