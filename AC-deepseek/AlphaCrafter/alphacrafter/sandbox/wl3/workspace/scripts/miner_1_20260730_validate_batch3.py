"""miner_1 2026-07-30: fast batch validation of NEW factor candidates.

Vectorized rank-IC (row-wise Spearman via masked Pearson on ranks) replaces the
slow per-date loop in factor_common.validate_factor -> ~1-2s per candidate.

Admission gate: |IC|>=0.007 and |ICIR|>=0.084 at h=10, plus
max_abs_library_correlation < 0.5 vs the 5-factor active library.
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel,
                           forward_returns, WATCHLIST, VAL_START, VAL_END)

t0 = time.time()
prices = load_prices(days=2200)
print(f'[load] {len(prices)} assets in {time.time()-t0:.1f}s', flush=True)
dxy = load_index('DXY', prices=prices)
usdjpy = load_index('USDJPY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
vix = load_index('VIX', prices=prices)
cn10y = prices.get('CN10Y')
xau = prices.get('XAU')
wti = prices.get('WTI')


def rank_ic_series_fast(factor_panel, fwd_ret, min_valid=8):
    """Vectorized daily cross-sectional Spearman IC between factor and fwd ret."""
    df = pd.concat({'x': factor_panel, 'y': fwd_ret}, axis=1).sort_index()
    x = df['x'].rank(axis=1)
    y = df['y'].rank(axis=1)
    valid = df['x'].notna() & df['y'].notna() & np.isfinite(df['x']) & np.isfinite(df['y'])
    n = valid.sum(axis=1)
    x = x.where(valid)
    y = y.where(valid)
    mx = x.mean(axis=1)
    my = y.mean(axis=1)
    cov = ((x.sub(mx, axis=0)) * (y.sub(my, axis=0))).sum(axis=1) / (n - 1)
    vx = ((x.sub(mx, axis=0)) ** 2).sum(axis=1) / (n - 1)
    vy = ((y.sub(my, axis=0)) ** 2).sum(axis=1) / (n - 1)
    ic = cov / np.sqrt(vx * vy)
    ic = ic.where((n >= min_valid) & (vx > 0) & (vy > 0))
    return ic.dropna()


def validate_fast(factor_panel, prices, horizons=(1, 2, 3, 5, 10, 20), min_valid=8):
    fwd = {h: forward_returns(prices, h) for h in horizons}
    ic_s = {h: rank_ic_series_fast(factor_panel, fwd[h], min_valid) for h in horizons}
    ic10 = ic_s[10]
    ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    if len(ic10) < 100:
        return None
    ic_mean = float(ic10.mean())
    ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    fac = factor_panel[(factor_panel.index >= VAL_START) & (factor_panel.index <= VAL_END)]
    total = fac.shape[0] * fac.shape[1]
    coverage = float(fac.notna().sum().sum()) / total if total else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= min_valid).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    decay = {str(h): (float(ic_s[h].mean()) if len(ic_s[h]) else float('nan')) for h in horizons}
    return {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit, 'n_ic_dates': int(len(ic10)),
            'coverage_asset_days': coverage, 'coverage_dates_ge8': ge8,
            'turnover_10d_rank': turn, 'decay_ic_by_horizon': decay}


def max_lib_corr_fast(factor_panel, lib_panels, min_valid=8):
    best, best_id = 0.0, None
    for fid, lp in lib_panels.items():
        if lp is None or len(lp) == 0:
            continue
        df = pd.concat({'x': factor_panel, 'y': lp}, axis=1)
        x = df['x'].rank(axis=1); y = df['y'].rank(axis=1)
        valid = df['x'].notna() & df['y'].notna() & np.isfinite(df['x']) & np.isfinite(df['y'])
        n = valid.sum(axis=1)
        x = x.where(valid); y = y.where(valid)
        mx = x.mean(axis=1); my = y.mean(axis=1)
        cov = ((x.sub(mx, axis=0)) * (y.sub(my, axis=0))).sum(axis=1) / (n - 1)
        vx = ((x.sub(mx, axis=0)) ** 2).sum(axis=1) / (n - 1)
        vy = ((y.sub(my, axis=0)) ** 2).sum(axis=1) / (n - 1)
        ic = cov / np.sqrt(vx * vy)
        ic = ic.where((n >= min_valid) & (vx > 0) & (vy > 0)).dropna()
        r = float(ic.mean())
        if np.isfinite(r) and abs(r) > best:
            best, best_id = abs(r), fid
    return best, best_id


# ---------- library panels: the 5 currently EFFECTIVE persisted factors ----------
def f_boll(df, s):
    m = df['close'].rolling(20).mean(); sd = df['close'].rolling(20).std()
    return (df['close'] - m) / sd.replace(0, np.nan)
def f_rsi(df, s):
    r = df['close'].pct_change()
    up = r.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-r.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)
def f_dxyb(df, s):
    if dxy is None: return None
    r = df['close'].pct_change(); dr = dxy['close'].pct_change()
    z = pd.concat([r.rename('r'), dr.rename('d')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var().replace(0, np.nan)
    move = dxy['close'] / dxy['close'].shift(20) - 1.0
    return (b * move).reindex(z.index)
def f_hilo(df, s):
    hi = df['high'].rolling(20).max(); lo = df['low'].rolling(20).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)
def f_vam(df, s):
    m = df['close'].shift(5) / df['close'].shift(25) - 1.0
    v = df['close'].pct_change().rolling(60).std().replace(0, np.nan)
    return m / v
LIB = {'bollinger_z_20d': f_boll, 'rsi_14d': f_rsi, 'dxy_beta_cond_60x20': f_dxyb,
       'high_low_range_pos_20': f_hilo, 'vol_adj_mom_20_60': f_vam}
lib_panels = {k: factor_to_panel(fn, prices) for k, fn in LIB.items()}
print('[lib] panels:', {k: v.shape for k, v in lib_panels.items()}, flush=True)

# ---------- candidate definitions ----------
def f_clv_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['close'] - df['low']) / rng).rolling(20).mean()
def f_down_mom_20(df, s):
    r = df['close'].pct_change()
    return r.clip(upper=0).rolling(20).mean()
def f_vol_exp_20x60(df, s):
    v20 = df['close'].pct_change().rolling(20).std()
    v60 = df['close'].pct_change().rolling(60).std()
    return v20 / v60.replace(0, np.nan) - 1.0
def f_trend_steep(df, s):
    c = df['close']
    return c / c.rolling(20).mean() - c / c.rolling(60).mean()
def f_stoch_k_20(df, s):
    lo = df['close'].rolling(20).min(); hi = df['close'].rolling(20).max()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)
def f_aroon_25(df, s):
    n = 25
    hi = df['high'].rolling(n, min_periods=n).apply(lambda x: n - 1 - x.argmax(), raw=True)
    lo = df['low'].rolling(n, min_periods=n).apply(lambda x: n - 1 - x.argmin(), raw=True)
    return (hi - lo) / n
def f_mom_accel(df, s):
    c = df['close']
    return (c.shift(5) / c.shift(25) - 1.0) - (c.shift(5) / c.shift(65) - 1.0)
def f_keltner_pos_20(df, s):
    c = df['close']
    sma = c.rolling(20).mean()
    tr = pd.concat([df['high'] - df['low'], (df['high'] - c.shift(1)).abs(),
                    (df['low'] - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20).mean().replace(0, np.nan)
    return (c - (sma - 2 * atr)) / (4 * atr)
def f_skew20_signed(df, s):
    r = df['close'].pct_change()
    sk = r.rolling(20).skew()
    m = df['close'] / df['close'].shift(20) - 1.0
    return sk * np.sign(m)
def f_ni_split_20(df, s):
    over = df['open'] / df['close'].shift(1) - 1.0
    intra = df['close'] / df['open'] - 1.0
    return (over - intra).rolling(20).mean()
def f_upper_wick_10(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    uw = (df['high'] - np.maximum(df['open'], df['close'])) / rng
    return uw.rolling(10).mean()
def f_pullback_20(df, s):
    c = df['close']
    hi = c.rolling(20).max()
    v = c.pct_change().rolling(20).std().replace(0, np.nan)
    return (c / hi - 1.0) / v
def f_vol_zscore(df, s):
    v = df['close'].pct_change().rolling(20).std()
    mu = v.rolling(60).mean(); sd = v.rolling(60).std()
    return (v - mu) / sd.replace(0, np.nan)
def f_skew_60(df, s):
    return df['close'].pct_change().rolling(60).skew()
def f_dd_60(df, s):
    hi = df['close'].rolling(60).max()
    return df['close'] / hi - 1.0
def f_eff_ratio_60(df, s):
    n = 60
    num = (df['close'] - df['close'].shift(n)).abs()
    den = df['close'].pct_change().abs().rolling(n).sum()
    return num / den.replace(0, np.nan)
def f_mom20_risk_adj(df, s):
    m = df['close'].shift(5) / df['close'].shift(25) - 1.0
    v = df['close'].pct_change().rolling(20).std().replace(0, np.nan)
    return m / v
def _beta_cond(df, ref, label):
    r = df['close'].pct_change(); rr = ref['close'].pct_change()
    z = pd.concat([r.rename('r'), rr.rename(label)], axis=1).dropna()
    b = z['r'].rolling(60).cov(z[label]) / z[label].rolling(60).var().replace(0, np.nan)
    move = ref['close'] / ref['close'].shift(20) - 1.0
    return (b * move).reindex(z.index)
def f_usdjpy_beta(df, s):
    return _beta_cond(df, usdjpy, 'u') if usdjpy is not None else None
def f_xau_beta(df, s):
    return _beta_cond(df, xau, 'x') if xau is not None else None
def f_wti_beta(df, s):
    return _beta_cond(df, wti, 'w') if wti is not None else None
def f_eurusd_beta(df, s):
    return _beta_cond(df, eurusd, 'e') if eurusd is not None else None
def f_vixbeta_level(df, s):
    return _beta_cond(df, vix, 'v') if vix is not None else None
def f_cn10y_beta(df, s):
    return _beta_cond(df, cn10y, 'c') if cn10y is not None else None
def f_max_ret_20(df, s):
    return df['close'].pct_change().rolling(20).max()
def f_upday_ratio_60(df, s):
    return (df['close'].pct_change() > 0).astype(float).rolling(60).mean() - 0.5
def f_bw_zscore(df, s):
    c = df['close']
    ma = c.rolling(20).mean(); sd = c.rolling(20).std()
    bw = 2.0 * sd / ma
    mu = bw.rolling(60).mean(); s = bw.rolling(60).std()
    return (bw - mu) / s.replace(0, np.nan)
def f_gk_ratio(df, s):
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    lh = np.log(h / o); ll = np.log(l / o); lc = np.log(c / o)
    gk = np.sqrt((0.5 * lh**2 - (2*np.log(2)-1) * lc**2 + lh * ll).clip(lower=0))
    cc = c.pct_change().rolling(20).std().replace(0, np.nan)
    return gk.rolling(20).mean() / cc
def f_skew_term(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()
def f_mom10(df, s): return df['close'].shift(5) / df['close'].shift(15) - 1.0
def f_mom120(df, s): return df['close'].shift(5) / df['close'].shift(125) - 1.0
def f_vixbeta(df, s):
    if vix is None: return None
    r = df['close'].pct_change(); vr = vix['close'].pct_change()
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var().replace(0, np.nan)
    return (-b * (vix['close'] / vix['close'].shift(20) - 1.0)).reindex(z.index)
def f_volvol(df, s): return df['close'].pct_change().rolling(20).std().rolling(60).std()

CANDIDATES = {
    'clv_20': f_clv_20, 'down_mom_20': f_down_mom_20, 'vol_exp_20x60': f_vol_exp_20x60,
    'trend_steep': f_trend_steep, 'stoch_k_20': f_stoch_k_20, 'aroon_25': f_aroon_25,
    'mom_accel_20_60': f_mom_accel, 'keltner_pos_20': f_keltner_pos_20,
    'skew20_signed': f_skew20_signed, 'ni_split_20': f_ni_split_20, 'upper_wick_10': f_upper_wick_10,
    'pullback_20_vol': f_pullback_20, 'vol_zscore_20x60': f_vol_zscore, 'skew_60d': f_skew_60,
    'dd_60d': f_dd_60, 'eff_ratio_60d': f_eff_ratio_60, 'mom20_risk_adj': f_mom20_risk_adj,
    'max_ret_20d': f_max_ret_20, 'upday_ratio_60': f_upday_ratio_60, 'bw_zscore_20_60': f_bw_zscore,
    'gk_vol_ratio_20': f_gk_ratio, 'skew_term_20_60': f_skew_term,
    'usdjpy_beta_cond_60x20': f_usdjpy_beta, 'xau_beta_cond_60x20': f_xau_beta,
    'wti_beta_cond_60x20': f_wti_beta, 'eurusd_beta_cond_60x20': f_eurusd_beta,
    'vixbeta_level_60x20': f_vixbeta_level, 'cn10y_beta_cond_60x20': f_cn10y_beta,
    'mom_10d_skip5': f_mom10, 'mom_120d_skip5': f_mom120,
    'vix_beta_cond_60x20': f_vixbeta, 'vol_of_vol20x60': f_volvol,
}

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 1
groups = {1: list(CANDIDATES)[:11], 2: list(CANDIDATES)[11:22], 3: list(CANDIDATES)[22:]}
ids = groups[BATCH]
print(f'[batch {BATCH}] {len(ids)} candidates', flush=True)

results = {}
for fid in ids:
    t = time.time()
    try:
        panel = factor_to_panel(CANDIDATES[fid], prices)
        m = validate_fast(panel, prices)
        if m is None:
            print(f'{fid:26s} insufficient data', flush=True)
            results[fid] = {'ok': False, 'reason': 'insufficient data'}
            continue
        rho, rho_id = max_lib_corr_fast(panel, lib_panels)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rho_id
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        rho_ok = (rho is not None) and (rho < 0.5)
        flag = 'PASS' if (ok and rho_ok) else 'FAIL'
        print(f'{fid:26s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} hit={m["ic_hit_ratio"]:.3f} '
              f'cov={m["coverage_asset_days"]:.2f} ge8={m["coverage_dates_ge8"]:.2f} '
              f'turn={m["turnover_10d_rank"]:.2f} rho={rho:.3f}({rho_id}) ndates={m["n_ic_dates"]} '
              f'[{time.time()-t:.1f}s] -> {flag}', flush=True)
        print('   decay:', {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}, flush=True)
        results[fid] = {'ok': ok and rho_ok, 'metrics': m}
    except Exception as e:
        print(f'{fid:26s} ERROR: {e}', flush=True)
        results[fid] = {'ok': False, 'reason': str(e)}

out = f'scripts/miner_1_20260730_results_batch{BATCH}.json'
with open(out, 'w') as fh:
    json.dump(results, fh, indent=1, default=str)
print(f'[saved] {out} | passed: {[k for k,v in results.items() if v.get("ok")]}', flush=True)
print(f'[total] {time.time()-t0:.1f}s', flush=True)
