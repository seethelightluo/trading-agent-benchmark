"""miner_1 2026-07-30: corrected drift re-validation + redundancy audit + restore/new candidates.

Library panels recomputed from persisted JSON definitions on one aligned grid.
Fully vectorized IC and pairwise-rho so it runs fast.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel,
                           WATCHLIST, VAL_START, VAL_END)

prices = load_prices(days=2200)
print(f'[load] {len(prices)} assets', flush=True)

FWD = {h: pd.DataFrame({s: df['close'].shift(-h) / df['close'] - 1.0
                        for s, df in prices.items()}).sort_index() for h in (1, 2, 3, 5, 10, 20)}

def rank_ic_series_fast(factor_panel, fwd_ret, min_valid=8):
    df = pd.concat({'x': factor_panel, 'y': fwd_ret}, axis=1, sort=True).sort_index()
    x = df['x'].rank(axis=1); y = df['y'].rank(axis=1)
    valid = df['x'].notna() & df['y'].notna() & np.isfinite(df['x']) & np.isfinite(df['y'])
    n = valid.sum(axis=1)
    x = x.where(valid); y = y.where(valid)
    mx = x.mean(axis=1); my = y.mean(axis=1)
    cov = ((x.sub(mx, axis=0)) * (y.sub(my, axis=0))).sum(axis=1) / (n - 1)
    vx = ((x.sub(mx, axis=0)) ** 2).sum(axis=1) / (n - 1)
    vy = ((y.sub(my, axis=0)) ** 2).sum(axis=1) / (n - 1)
    ic = cov / np.sqrt(vx * vy)
    return ic.where((n >= min_valid) & (vx > 0) & (vy > 0)).dropna()

def quick_ic(fac, h=10, start=VAL_START, end=VAL_END, min_valid=8):
    ic = rank_ic_series_fast(fac, FWD[h], min_valid)
    ic = ic[(ic.index >= start) & (ic.index <= end)]
    if len(ic) < 60:
        return None
    m = float(ic.mean()); s = float(ic.std(ddof=1))
    return {'ic': m, 'icir': m / s if s > 0 else 0.0, 'n': int(len(ic))}

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

# ---------------- library factor definitions ----------------
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
def f_max_ret(df, s):
    return df['close'].pct_change().rolling(20).max()
def f_mom10(df, s): return df['close'].shift(5) / df['close'].shift(15) - 1.0
def f_mom120(df, s): return df['close'].shift(5) / df['close'].shift(125) - 1.0
def f_rsi14(df, s):
    d = df['close'].diff()
    up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1 / 14, adjust=False).mean()
    rd = dn.ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))
def f_skew_term(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()
def f_spx_beta(df, s):
    spx = prices.get('SPX'); r = df['close'].pct_change(); rs = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()).reindex(z.index)
def f_upper_wick(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['high'] - np.maximum(df['open'], df['close'])) / rng).rolling(10).mean()
def f_vix_cond(df, s):
    if vix is None: return None
    r = df['close'].pct_change(); rv = vix['close'].pct_change()
    z = pd.concat([r.rename('r'), rv.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
    return (-b * (vix['close'] / vix['close'].shift(20) - 1.0)).reindex(z.index)
def f_vol_adj_mom(df, s):
    m = df['close'].shift(5) / df['close'].shift(25) - 1.0
    v = df['close'].pct_change().rolling(60).std()
    return m / v.replace(0, np.nan)
def f_volvol(df, s): return df['close'].pct_change().rolling(20).std().rolling(60).std()
def f_vol_price_corr(df, s):
    return df['close'].pct_change().rolling(60).corr(df['volume']).replace([np.inf, -np.inf], np.nan)
def f_wti_beta(df, s):
    wti = prices.get('WTI'); r = df['close'].pct_change(); rw = wti['close'].pct_change()
    z = pd.concat([r.rename('r'), rw.rename('w')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['w']) / z['w'].rolling(60).var()).reindex(z.index)

dxy = load_index('DXY', prices=prices)
vix = load_index('VIX', prices=prices)

LIB = {
    'bollinger_z_20d': f_bollinger_z, 'brkout_60_20': f_brkout, 'btc_beta_60': f_btc_beta,
    'bw_zscore_20_60': f_bw_zscore, 'dxy_beta_cond_60x20': f_dxy_cond,
    'high_low_range_pos_20': f_hilo20, 'hs300_beta_60': f_hs300_beta, 'max_ret_20d': f_max_ret,
    'mom_10d_skip5': f_mom10, 'mom_120d_skip5': f_mom120, 'rsi_14d': f_rsi14,
    'skew_term_20_60': f_skew_term, 'spx_beta_60': f_spx_beta, 'upper_wick_10': f_upper_wick,
    'vix_beta_cond_60x20': f_vix_cond, 'vol_adj_mom_20_60': f_vol_adj_mom,
    'vol_of_vol20x60': f_volvol, 'vol_price_corr_60': f_vol_price_corr, 'wti_beta_60': f_wti_beta,
}

print('computing library panels...', flush=True)
lib_panels = {}
for fid, fn in LIB.items():
    try:
        p = factor_to_panel(fn, prices)
        if p.shape[0] > 100:
            lib_panels[fid] = p
        else:
            print(f'  {fid}: panel too small {p.shape}', flush=True)
    except Exception as e:
        print(f'  {fid}: ERR {e}', flush=True)

RECENT = pd.Timestamp('2025-07-01')
print('\n=== A. DRIFT (full vs recent 1y) ===')
for fid, p in sorted(lib_panels.items()):
    mf = quick_ic(p)
    mr = quick_ic(p, start=RECENT, end=VAL_END)
    if mf is None or mr is None:
        print(f'  {fid:26s} insufficient', flush=True); continue
    flag = '  <-- DRIFT' if (abs(mr['ic']) < 0.007 or abs(mr['icir']) < 0.084) else ''
    print(f'  {fid:26s} FULL IC={mf["ic"]:+.4f} ICIR={mf["icir"]:+.4f} | RECENT IC={mr["ic"]:+.4f} ICIR={mr["icir"]:+.4f}{flag}', flush=True)

print('\n=== B. PAIRWISE |rho| >= 0.5 ===')
fids = sorted(lib_panels)
for i in range(len(fids)):
    for j in range(i + 1, len(fids)):
        r = pair_rho(lib_panels[fids[i]], lib_panels[fids[j]])
        if abs(r) >= 0.5:
            print(f'  {fids[i]:26s} <-> {fids[j]:26s} rho={r:+.3f}', flush=True)

def max_lib_corr(panel, min_valid=8):
    best, best_id = 0.0, None
    for fid, lp in lib_panels.items():
        r = pair_rho(panel, lp, min_valid)
        if np.isfinite(r) and abs(r) > best:
            best, best_id = abs(r), fid
    return best, best_id

us10y = prices.get('US10Y'); cn10y = prices.get('CN10Y')
def f_rates_spread_cond(df, s):
    if us10y is None or cn10y is None: return None
    spr = us10y['close'] - cn10y['close']
    r = df['close'].pct_change(); rs = spr.pct_change()
    z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()
    return (b * (spr / spr.shift(20) - 1.0)).reindex(z.index)
def f_downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    neg = r.where(r < 0, 0.0)
    sv = r.rolling(60).std(); sd = neg.rolling(60).std()
    return sd / sv.replace(0, np.nan)
def f_trend_r2_60(df, s):
    t = pd.Series(np.arange(len(df)), index=df.index)
    return df['close'].rolling(60).corr(t).replace([np.inf, -np.inf], np.nan)

CAND = [
    ('brkout_60_20_restore', f_brkout, 'restore (had no artifact)'),
    ('rates_spread_beta_cond_60x20', f_rates_spread_cond, 'US10Y-CN10Y spread beta x 20d move'),
    ('downside_vol_ratio_60', f_downside_vol_ratio_60, 'downside semi-vol ratio 60d'),
    ('trend_r2_60', f_trend_r2_60, 'trend consistency corr(close,time) 60d'),
]

print('\n=== C. CANDIDATES ===')
for fid, fn, idea in CAND:
    p = factor_to_panel(fn, prices)
    mf = quick_ic(p)
    if mf is None:
        print(f'  {fid:30s} insufficient', flush=True); continue
    rho, rho_id = max_lib_corr(p)
    ok = abs(mf['ic']) >= 0.007 and abs(mf['icir']) >= 0.084 and rho < 0.5
    print(f'  {fid:30s} IC={mf["ic"]:+.4f} ICIR={mf["icir"]:+.4f} rho={rho:.3f}({rho_id}) -> {"PASS" if ok else "FAIL"}  [{idea}]', flush=True)

print('\nDONE')
