"""miner_1 2026-07-30: post-deprecation pairwise audit + refresh max_ret_20d provenance rho.

After deprecating vol_of_vol20x60 / upper_wick_10 / vix_beta_cond_60x20, recompute
the pairwise rho among remaining EFFECTIVE artifact factors and update stale
self-reported max_abs_library_correlation in JSONs (provenance only; the
deterministic gate recomputes from artifacts anyway).
"""
import sys, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, WATCHLIST

prices = load_prices(days=2200)

def pair_rho(p1, p2, min_valid=8):
    idx = p1.index.intersection(p2.index)
    x = p1.loc[idx].rank(axis=1); y = p2.loc[idx].rank(axis=1)
    valid = p1.loc[idx].notna() & p2.loc[idx].notna() & np.isfinite(p1.loc[idx]) & np.isfinite(p2.loc[idx])
    n = valid.sum(axis=1)
    x = x.where(valid); y = y.where(valid)
    mx = x.mean(axis=1); my = y.mean(axis=1)
    cov = ((x.sub(mx, axis=0)) * (y.sub(my, axis=0))).sum(axis=1) / (n - 1)
    vx = ((x.sub(mx, axis=0)) ** 2).sum(axis=1) / (n - 1)
    vy = ((y.sub(my, axis=0)) ** 2).sum(axis=1) / (n - 1)
    ic = cov / np.sqrt(vx * vy)
    vals = ic[(n >= min_valid) & (vx > 0) & (vy > 0)].dropna()
    return float(vals.mean()) if len(vals) else np.nan

def f_bollinger_z(df, s):
    m = df['close'].rolling(20).mean(); sd = df['close'].rolling(20).std(ddof=1)
    return (df['close'] - m) / sd.replace(0, np.nan)
def f_brkout(df, s):
    hi = df['high'].rolling(60).max().shift(1)
    return (df['close'] > hi).astype(float).rolling(20).mean()
def f_btc_beta(df, s):
    btc = prices.get('BTC'); r = df['close'].pct_change(); rb = btc['close'].pct_change()
    z = pd.concat([r.rename('r'), rb.rename('b')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var()).reindex(z.index)
def f_bw_zscore(df, s):
    c = df['close']; ma = c.rolling(20).mean(); sd = c.rolling(20).std()
    bw = 2.0 * sd / ma.replace(0, np.nan)
    mu = bw.rolling(60).mean(); ss = bw.rolling(60).std()
    return (bw - mu) / ss.replace(0, np.nan)
def f_dxy_cond(df, s):
    dxy = None
    try:
        from factor_common import load_index
        dxy = load_index('DXY', prices=prices)
    except Exception:
        pass
    if dxy is None: return None
    r = df['close'].pct_change(); rd = dxy['close'].pct_change()
    z = pd.concat([r.rename('r'), rd.rename('d')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()
    return (b * (dxy['close'] / dxy['close'].shift(20) - 1.0)).reindex(z.index)
def f_hilo20(df, s):
    hi = df['high'].rolling(20).max(); lo = df['low'].rolling(20).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)
def f_hs300_beta(df, s):
    a = prices.get('000300.SH'); r = df['close'].pct_change(); ra = a['close'].pct_change()
    z = pd.concat([r.rename('r'), ra.rename('a')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var()).reindex(z.index)
def f_max_ret(df, s): return df['close'].pct_change().rolling(20).max()
def f_mom10(df, s): return df['close'].shift(5) / df['close'].shift(15) - 1.0
def f_mom120(df, s): return df['close'].shift(5) / df['close'].shift(125) - 1.0
def f_rsi14(df, s):
    d = df['close'].diff()
    up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1 / 14, adjust=False).mean(); rd = dn.ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))
def f_skew_term(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()
def f_spx_beta(df, s):
    spx = prices.get('SPX'); r = df['close'].pct_change(); rs = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()).reindex(z.index)
def f_vol_price_corr(df, s):
    return df['close'].pct_change().rolling(60).corr(df['volume']).replace([np.inf, -np.inf], np.nan)
def f_wti_beta(df, s):
    wti = prices.get('WTI'); r = df['close'].pct_change(); rw = wti['close'].pct_change()
    z = pd.concat([r.rename('r'), rw.rename('w')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['w']) / z['w'].rolling(60).var()).reindex(z.index)

DEFS = {'bollinger_z_20d': f_bollinger_z, 'brkout_60_20': f_brkout, 'btc_beta_60': f_btc_beta,
        'bw_zscore_20_60': f_bw_zscore, 'dxy_beta_cond_60x20': f_dxy_cond,
        'high_low_range_pos_20': f_hilo20, 'hs300_beta_60': f_hs300_beta, 'max_ret_20d': f_max_ret,
        'mom_10d_skip5': f_mom10, 'mom_120d_skip5': f_mom120, 'rsi_14d': f_rsi14,
        'skew_term_20_60': f_skew_term, 'spx_beta_60': f_spx_beta,
        'vol_price_corr_60': f_vol_price_corr, 'wti_beta_60': f_wti_beta}

# effective = artifact-bearing, non-deprecated
effective = set()
for p in glob.glob('factors/*.json'):
    if 'ensemble' in p or 'deprecated' in p:
        continue
    d = json.load(open(p))
    if d.get('signal_artifact') and d['validation']['status'] == 'EFFECTIVE':
        effective.add(d['factor_id'])

print('effective set:', sorted(effective), flush=True)
panels = {fid: factor_to_panel(fn, prices) for fid, fn in DEFS.items() if fid in effective}
panels = {fid: p for fid, p in panels.items() if p.shape[0] > 100}

print('\n=== pairwise |rho| >= 0.5 among post-deprecation effective set ===')
fids = sorted(panels)
for i in range(len(fids)):
    for j in range(i + 1, len(fids)):
        r = pair_rho(panels[fids[i]], panels[fids[j]])
        if abs(r) >= 0.5:
            print(f'  {fids[i]:26s} <-> {fids[j]:26s} rho={r:+.3f}', flush=True)

print('\n=== refresh self-reported rho in JSONs (provenance) ===')
for fid, p in sorted(panels.items()):
    best, best_id = 0.0, None
    for fid2, p2 in panels.items():
        if fid2 == fid:
            continue
        r = pair_rho(p, p2)
        if np.isfinite(r) and abs(r) > best:
            best, best_id = abs(r), fid2
    jp = f'factors/{fid}.json'
    d = json.load(open(jp))
    old = d['validation']['metrics'].get('max_abs_library_correlation')
    d['validation']['metrics']['max_abs_library_correlation'] = best
    d['validation']['metrics']['max_corr_library_id'] = best_id
    json.dump(d, open(jp, 'w'), indent=2, default=str)
    print(f'  {fid:26s} rho {old if old is not None else "None"} -> {best:.3f} (vs {best_id})', flush=True)

print('DONE')
