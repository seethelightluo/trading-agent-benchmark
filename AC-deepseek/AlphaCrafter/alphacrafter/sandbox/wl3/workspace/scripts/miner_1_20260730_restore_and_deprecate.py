"""miner_1 2026-07-30: restore ghost factors with signal artifacts + final candidate batch + deprecate drifted.

A. Ghost factors (EFFECTIVE but no artifact in factors/ root): brkout_60_20,
   hilo_pos_60, mom_10d_skip5 — restore with .npy artifact IF they still pass
   |IC|>=0.007 & |ICIR|>=0.084 and rho<0.5 vs the artifact-effective set.
B. New orthogonal candidates.
C. Deprecate factors whose RECENT-1y re-validation fails badly (negative/near-zero
   ICIR): vix_beta_cond_60x20 (fails full-window too), upper_wick_10,
   vol_of_vol20x60 (recent ICIR<0) -> rename *_deprecated.json, status DEPRECATED.
"""
import sys, json, os, glob
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel, persist_factor,
                           canonical_grid, WATCHLIST, VAL_START, VAL_END)

prices = load_prices(days=2200)
grid = canonical_grid(prices)
print(f'[load] {len(prices)} assets; grid n={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

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

def validate_fast(fac, prices_, horizons=(1, 2, 3, 5, 10, 20), min_valid=8):
    ic_s = {h: rank_ic_series_fast(fac, FWD[h], min_valid) for h in horizons}
    ic10 = ic_s[10][(ic_s[10].index >= VAL_START) & (ic_s[10].index <= VAL_END)]
    if len(ic10) < 100:
        return None
    m = float(ic10.mean()); s = float(ic10.std(ddof=1))
    fac2 = fac[(fac.index >= VAL_START) & (fac.index <= VAL_END)]
    total = fac2.shape[0] * fac2.shape[1]
    coverage = float(fac2.notna().sum().sum()) / total if total else 0.0
    ge8 = float((fac2.notna().sum(axis=1) >= min_valid).mean())
    ranked = fac2.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    return {'ic': float(m), 'icir': float(m / s if s > 0 else 0.0),
            'ic_hit_ratio': float((ic10 > 0).mean()) if m >= 0 else float((ic10 < 0).mean()),
            'n_ic_dates': int(len(ic10)), 'coverage_asset_days': coverage,
            'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
            'decay_ic_by_horizon': {str(h): (float(ic_s[h].mean()) if len(ic_s[h]) else float('nan')) for h in horizons}}

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

# ---------------- library definitions (same as drift audit) ----------------
def f_brkout(df, s):
    hi = df['high'].rolling(60).max().shift(1)
    return (df['close'] > hi).astype(float).rolling(20).mean()
def f_mom10(df, s): return df['close'].shift(5) / df['close'].shift(15) - 1.0
def f_hilo60(df, s):
    hi = df['high'].rolling(60).max(); lo = df['low'].rolling(60).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)
def f_bollinger_z(df, s):
    m = df['close'].rolling(20).mean(); sd = df['close'].rolling(20).std(ddof=1)
    return (df['close'] - m) / sd.replace(0, np.nan)
def f_rsi14(df, s):
    d = df['close'].diff()
    up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1 / 14, adjust=False).mean(); rd = dn.ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))
def f_hilo20(df, s):
    hi = df['high'].rolling(20).max(); lo = df['low'].rolling(20).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)
def f_mom120(df, s): return df['close'].shift(5) / df['close'].shift(125) - 1.0
def f_bw_zscore(df, s):
    c = df['close']; ma = c.rolling(20).mean(); sd = c.rolling(20).std()
    bw = 2.0 * sd / ma.replace(0, np.nan)
    mu = bw.rolling(60).mean(); ss = bw.rolling(60).std()
    return (bw - mu) / ss.replace(0, np.nan)
def f_max_ret(df, s): return df['close'].pct_change().rolling(20).max()
def f_skew_term(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()
def f_upper_wick(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['high'] - np.maximum(df['open'], df['close'])) / rng).rolling(10).mean()
def f_volvol(df, s): return df['close'].pct_change().rolling(20).std().rolling(60).std()
def f_vol_price_corr(df, s):
    return df['close'].pct_change().rolling(60).corr(df['volume']).replace([np.inf, -np.inf], np.nan)
def f_btc_beta(df, s):
    btc = prices.get('BTC'); r = df['close'].pct_change(); rb = btc['close'].pct_change()
    z = pd.concat([r.rename('r'), rb.rename('b')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var()).reindex(z.index)
def f_spx_beta(df, s):
    spx = prices.get('SPX'); r = df['close'].pct_change(); rs = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()).reindex(z.index)
def f_wti_beta(df, s):
    wti = prices.get('WTI'); r = df['close'].pct_change(); rw = wti['close'].pct_change()
    z = pd.concat([r.rename('r'), rw.rename('w')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['w']) / z['w'].rolling(60).var()).reindex(z.index)
def f_dxy_cond(df, s):
    if dxy is None: return None
    r = df['close'].pct_change(); rd = dxy['close'].pct_change()
    z = pd.concat([r.rename('r'), rd.rename('d')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()
    return (b * (dxy['close'] / dxy['close'].shift(20) - 1.0)).reindex(z.index)
def f_hs300_beta(df, s):
    a = prices.get('000300.SH'); r = df['close'].pct_change(); ra = a['close'].pct_change()
    z = pd.concat([r.rename('r'), ra.rename('a')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var()).reindex(z.index)

dxy = load_index('DXY', prices=prices)

# artifact-effective set (has .npy in factors/)
ARTIFACT_IDS = set()
for p in glob.glob('factors/*.json'):
    if 'ensemble' in p or 'deprecated' in p:
        continue
    d = json.load(open(p))
    if d.get('signal_artifact'):
        ARTIFACT_IDS.add(d['factor_id'])

ALL_DEFS = {'brkout_60_20': f_brkout, 'mom_10d_skip5': f_mom10, 'hilo_pos_60': f_hilo60,
            'bollinger_z_20d': f_bollinger_z, 'rsi_14d': f_rsi14,
            'high_low_range_pos_20': f_hilo20, 'mom_120d_skip5': f_mom120,
            'bw_zscore_20_60': f_bw_zscore, 'max_ret_20d': f_max_ret,
            'skew_term_20_60': f_skew_term, 'upper_wick_10': f_upper_wick,
            'vol_of_vol20x60': f_volvol, 'vol_price_corr_60': f_vol_price_corr,
            'btc_beta_60': f_btc_beta, 'spx_beta_60': f_spx_beta, 'wti_beta_60': f_wti_beta,
            'dxy_beta_cond_60x20': f_dxy_cond, 'hs300_beta_60': f_hs300_beta}

print('artifact-effective ids:', sorted(ARTIFACT_IDS), flush=True)
lib_effective = {fid: factor_to_panel(fn, prices) for fid, fn in ALL_DEFS.items() if fid in ARTIFACT_IDS}
lib_effective = {fid: p for fid, p in lib_effective.items() if p.shape[0] > 100}
print('effective panels computed:', sorted(lib_effective), flush=True)

def max_lib_corr(panel, exclude=()):
    best, best_id = 0.0, None
    for fid, lp in lib_effective.items():
        if fid in exclude:
            continue
        r = pair_rho(panel, lp)
        if np.isfinite(r) and abs(r) > best:
            best, best_id = abs(r), fid
    return best, best_id

# ============ A. RESTORE GHOST FACTORS ============
RESTORES = [
    ('brkout_60_20', f_brkout, 'Breakout 60d/20d',
     'mean20(close > shift1(max_high_60))',
     'Fraction of days over the past 20d on which close broke above the prior 60d high. '
     'Trend-breakout persistence signal; quarantined earlier solely for a missing signal '
     'artifact; restored with artifact. Recent-1y IC remains strong (+0.13).',
     ['close', 'high'], {'lookback': 60, 'window': 20}, 1,
     ['breakout', 'trend', 'restore']),
    ('mom_10d_skip5', f_mom10, 'Momentum 10d skip5',
     'close.shift(5)/close.shift(15) - 1',
     '10-day price momentum skipping the most recent 5 days (library factor restored with '
     'signal artifact after quarantine for missing artifact).',
     ['close'], {'lookback': 10, 'skip': 5}, 1,
     ['momentum', 'trend', 'restore']),
    ('hilo_pos_60', f_hilo60, 'Hi-Lo Position 60d',
     '(close - min(low,60)) / (max(high,60) - min(low,60))',
     'Position of close within the trailing 60-day high-low range. Restored with artifact '
     'if pairwise rho vs effective library stays < 0.5.',
     ['close', 'high', 'low'], {'window': 60}, 1,
     ['range-position', 'trend', 'restore']),
]

print('\n=== A. RESTORE GHOST FACTORS ===')
restored = []
for fid, fn, name, expr, desc, deps, params, dirn, tags in RESTORES:
    panel = factor_to_panel(fn, prices)
    m = validate_fast(panel, prices)
    if m is None:
        print(f'  {fid}: insufficient', flush=True); continue
    rho, rho_id = max_lib_corr(panel, exclude={fid})
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    print(f'  {fid:20s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} rho={rho:.3f}({rho_id}) -> {"PASS" if ok else "FAIL"}', flush=True)
    if not ok:
        continue
    path, arr = persist_factor(factor_id=fid, factor_name=name, expression=expr,
                               description=desc, dependencies=deps, parameters=params,
                               expected_direction=dirn, panel=panel, metrics=m, tags=tags,
                               grid=grid, prices=prices, version='2.0.0', status='EFFECTIVE',
                               regime_notes='Restored with artifact after quarantine for missing artifact; '
                                            'recent-1y drift check passes.')
    print(f'  wrote {path} art={arr.shape}', flush=True)
    restored.append(fid)

# ============ B. NEW CANDIDATES ============
hsi = prices.get('HSI')
def f_hsi_beta_cond(df, s):
    if hsi is None: return None
    r = df['close'].pct_change(); rh = hsi['close'].pct_change()
    z = pd.concat([r.rename('r'), rh.rename('h')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['h']) / z['h'].rolling(60).var()
    return (b * (hsi['close'] / hsi['close'].shift(20) - 1.0)).reindex(z.index)
def f_skew20_signed_mom60(df, s):
    r = df['close'].pct_change()
    sk = r.rolling(20).skew()
    m60 = df['close'].shift(5) / df['close'].shift(65) - 1.0
    return sk * np.sign(m60)
def f_vol_ratio_min60(df, s):
    v = df['close'].pct_change().rolling(20).std()
    mn = v.rolling(60).min()
    return v / mn.replace(0, np.nan)
def f_dd_ratio_60(df, s):
    # max cumulative drawup / |max drawdown| over 60d
    c = df['close']
    roll_max = c.rolling(60).max(); roll_min = c.rolling(60).min()
    du = (c / roll_max - 1.0)  # <=0 drawdown from peak; use drawup = c/shift-runmax - 1
    dd = (c / roll_min - 1.0)   # >=0 drawup from trough
    return dd.abs() / du.abs().replace(0, np.nan)

CAND = [
    ('hsi_beta_cond_60x20', f_hsi_beta_cond, 'HSI (Asia risk) beta x 20d move'),
    ('skew20_signed_mom60', f_skew20_signed_mom60, 'skew20 signed by 60d trend direction'),
    ('vol_ratio_min60', f_vol_ratio_min60, 'vol20 / min(vol20,60d) expansion ratio'),
    ('dd_ratio_60', f_dd_ratio_60, 'drawup/drawdown asymmetry 60d'),
]
print('\n=== B. NEW CANDIDATES ===')
for fid, fn, idea in CAND:
    panel = factor_to_panel(fn, prices)
    m = validate_fast(panel, prices)
    if m is None:
        print(f'  {fid:28s} insufficient', flush=True); continue
    rho, rho_id = max_lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    print(f'  {fid:28s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} rho={rho:.3f}({rho_id}) -> {"PASS" if ok else "FAIL"}  [{idea}]', flush=True)
    if ok:
        path, arr = persist_factor(factor_id=fid, factor_name=idea, expression='see description',
                                   description=idea, dependencies=['close', 'high', 'low', 'open'],
                                   parameters={}, expected_direction=1, panel=panel, metrics=m,
                                   tags=['new'], grid=grid, prices=prices, version='1.0.0',
                                   status='EFFECTIVE', regime_notes='New candidate 2026-07-30 warm-up validation.')
        restored.append(fid)
        print(f'  wrote {path} art={arr.shape}', flush=True)

# ============ C. DEPRECATE DRIFTED ============
print('\n=== C. DEPRECATION ===')
DEPRECATE = ['vix_beta_cond_60x20', 'upper_wick_10', 'vol_of_vol20x60']
for fid in DEPRECATE:
    src = f'factors/{fid}.json'
    dst = f'factors/{fid}_deprecated.json'
    if not os.path.exists(src):
        print(f'  {fid}: not in root, skip', flush=True); continue
    d = json.load(open(src))
    d['validation']['status'] = 'DEPRECATED'
    d['validation']['last_validated'] = '2026-07-30'
    d['validation']['regime_notes'] = (
        d.get('validation', {}).get('regime_notes', '') +
        ' | DEPRECATED 2026-07-30: recent-1y (2025-07..2026-07) re-validation failed - '
        'IC/ICIR below admission thresholds or negative ICIR (drift).')
    json.dump(d, open(dst, 'w'), indent=2, default=str)
    os.remove(src)
    print(f'  {fid} -> {dst} (status DEPRECATED)', flush=True)

# ============ READ-BACK VERIFICATION ============
print('\n=== READ-BACK VERIFICATION ===')
for fid in restored:
    p = f'factors/{fid}.json'
    d = json.load(open(p))
    art = np.load(f'factors/{fid}_signal.npy', allow_pickle=False)
    m = d['validation']['metrics']
    assert d['factor_id'] == fid, 'id mismatch'
    assert d['validation']['status'] == 'EFFECTIVE', 'status mismatch'
    assert abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084, 'gate mismatch'
    assert d.get('signal_artifact') is not None and art.shape == (len(grid), 15), 'artifact missing'
    print(f'  OK {fid}: status={d["validation"]["status"]} IC={m["ic"]:.4f} ICIR={m["icir"]:.4f} '
          f'rho={m.get("max_abs_library_correlation"):.3f} art={art.shape} '
          f'last_validated={d["validation"]["last_validated"]}', flush=True)

print('\nRESTORED/PERSISTED:', restored)
print('DONE')
