# -*- coding: utf-8 -*-
"""miner_1 2028-03-23: explore new factor batch.
Visible data through 2028-03-22 (current_date 2028-03-23).
Admission gates on 15-asset universe at 10d horizon: |IC| >= 0.0070 and
|ICIR| >= 0.0840; post-gate max_abs_library_correlation < 0.5.

Ideas (orthogonal to library: trend_r2, semi_down_ratio, mom 10/120d,
time_under_water, vol_of_vol, dxy_beta, WTI_BETA, vix_beta_cond, kurt_20,
tail_ratio_20):
  1. range_squeeze_20_60     volatility compression (20d/60d mean range)
  2. gap_ratio_20            overnight gap activity vs intraday range
  3. body_balance_20         signed candle body balance (bullish vs bearish)
  4. upper_shadow_ratio_20   selling pressure at highs
  5. lower_shadow_ratio_20   dip-buying support at lows
  6. down_vol_ratio_5_60     recent downside risk build-up
  7. maxmin_ratio_20         asymmetry of extreme daily gains vs losses
  8. corr_spx_60             systematic linkage to SPX (self = NaN)
  9. corr_wti_60             commodity linkage to WTI (self = NaN)
 10. mom_sign_consistency_60 trend participation (sign agreement with 60d mom)
 11. vol_zscore_20_252       20d vol vs its 1y history (vol spike/compression)
 12. stoch_pos_20            stochastic position in 20d high-low range
"""
import sys, json, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

# refresh library factor list from live factor files (EFFECTIVE + artifact)
lib_factors = []
for p in sorted(os.listdir('factors')):
    if not p.endswith('.json') or p == 'factor_ensemble.json':
        continue
    try:
        d = json.load(open('factors/' + p))
        if d.get('validation', {}).get('status') == 'EFFECTIVE' and \
           d.get('validation', {}).get('signal_artifact'):
            lib_factors.append(d['factor_id'])
    except Exception:
        pass
L.LIB_FACTORS = lib_factors
print('Library factors for rho check (%d): %s' % (len(lib_factors), lib_factors))

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets'
      % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))

WATCH = list(C.columns)


def safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)


def rolling_corr_with(asset_ret, bench_ret, window, bench_name):
    """Rolling Pearson correlation of each asset's returns with a benchmark.
    Reference asset itself is set to NaN (mechanical 1.0 otherwise)."""
    out = pd.DataFrame(index=asset_ret.index, columns=asset_ret.columns, dtype=float)
    for s in asset_ret.columns:
        if s == bench_name:
            out[s] = np.nan
            continue
        out[s] = asset_ret[s].rolling(window).corr(bench_ret)
    return out


def build(name):
    if name == 'range_squeeze_20_60':
        rng = (H - Lw)
        return safe_div(rng.rolling(20).mean(), rng.rolling(60).mean())
    if name == 'gap_ratio_20':
        prev_c = C.shift(1)
        gap = (O - prev_c).abs()
        rng = (H - Lw).replace(0, np.nan)
        return safe_div(gap, rng).rolling(20).mean()
    if name == 'body_balance_20':
        rng = (H - Lw).replace(0, np.nan)
        body = (C - O) / rng
        return body.rolling(20).sum()
    if name == 'upper_shadow_ratio_20':
        rng = (H - Lw).replace(0, np.nan)
        us = (H - np.maximum(C, O)) / rng
        return us.rolling(20).mean()
    if name == 'lower_shadow_ratio_20':
        rng = (H - Lw).replace(0, np.nan)
        ls = (np.minimum(C, O) - Lw) / rng
        return ls.rolling(20).mean()
    if name == 'down_vol_ratio_5_60':
        dn = R.where(R < 0)
        v5 = dn.rolling(5).std()
        v60 = dn.rolling(60).std()
        return safe_div(v5, v60)
    if name == 'maxmin_ratio_20':
        mx = R.rolling(20).max()
        mn = R.rolling(20).min().abs()
        return safe_div(mx, mn)
    if name == 'corr_spx_60':
        return rolling_corr_with(R, R['SPX'], 60, 'SPX')
    if name == 'corr_wti_60':
        return rolling_corr_with(R, R['WTI'], 60, 'WTI')
    if name == 'mom_sign_consistency_60':
        mom60 = (C / C.shift(60) - 1.0)
        sign_mom = np.sign(mom60)
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            agree = (np.sign(R[s]) == sign_mom[s]).astype(float)
            out[s] = agree.rolling(60).mean()
        return out
    if name == 'vol_zscore_20_252':
        vol20 = R.rolling(20).std()
        mu = vol20.rolling(252).mean()
        sd = vol20.rolling(252).std()
        return safe_div(vol20 - mu, sd)
    if name == 'stoch_pos_20':
        hi = H.rolling(20).max()
        lo = Lw.rolling(20).min()
        rng = (hi - lo).replace(0, np.nan)
        return safe_div(C - lo, rng)
    return None


CANDIDATES = ['range_squeeze_20_60', 'gap_ratio_20', 'body_balance_20',
              'upper_shadow_ratio_20', 'lower_shadow_ratio_20',
              'down_vol_ratio_5_60', 'maxmin_ratio_20', 'corr_spx_60',
              'corr_wti_60', 'mom_sign_consistency_60', 'vol_zscore_20_252',
              'stoch_pos_20']

out = {'visible_through': str(C.index.max().date()), 'n_dates': len(C),
       'n_assets': C.shape[1], 'library_factors': lib_factors, 'results': {}}

for name in CANDIDATES:
    try:
        fp = build(name)
        if fp is None:
            print('\n[%s] build failed' % name)
            continue
        summ = L.full_validate(fp, R, horizon=10, label=name)
    except Exception as e:
        print('\n[%s] validation error: %s' % (name, e))
        continue
    gate = (abs(summ['ic']) >= 0.007) and (abs(summ['icir']) >= 0.084)
    rho_ok = summ['max_abs_library_correlation'] < 0.5
    summ['pass_gate'] = bool(gate)
    summ['rho_ok'] = bool(rho_ok)
    # recency drift: IC over trailing 2y
    s_ic = None
    try:
        fr = R.shift(-10)
        from miner3_lib import rank_ic
        s_ic = rank_ic(fp, fr)
    except Exception:
        pass
    recent = None
    if s_ic is not None and len(s_ic) >= 20:
        sub = s_ic[s_ic.index >= '2026-03-23']
        if len(sub) >= 20:
            recent = {'ic': round(float(sub.mean()), 4),
                      'icir': round(float(sub.mean() / sub.std()), 4) if sub.std() > 0 else 0.0,
                      'n': int(len(sub))}
    summ['recency_2y'] = recent
    out['results'][name] = {k: summ[k] for k in
                            ['label', 'horizon', 'ic', 'icir', 'ic_hit_ratio',
                             'n_ic_dates', 'regime', 'coverage_asset_days',
                             'coverage_dates_ge8', 'turnover_10d_rank',
                             'decay_ic_by_horizon', 'max_abs_library_correlation',
                             'pass_gate', 'rho_ok', 'recency_2y']}
    print('\n=== %s ===' % name)
    print('  IC=%.4f ICIR=%.4f hit=%.3f n=%d cov_asset=%.3f cov_dates_ge8=%.3f turn=%.3f'
          % (summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
             summ['coverage_asset_days'], summ['coverage_dates_ge8'],
             summ['turnover_10d_rank']))
    print('  decay:', summ['decay_ic_by_horizon'])
    print('  regime:', {k: v['ic'] for k, v in summ.get('regime', {}).items()})
    print('  recency_2y:', recent)
    print('  max_abs_library_corr=%.3f' % summ['max_abs_library_correlation'])
    print('  GATE PASS: %s (ic=%s icir=%s rho_ok=%s)'
          % (gate and rho_ok, gate, abs(summ['icir']) >= 0.084, rho_ok))

with open('scripts/miner_1_20280323_explore_results.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print('\nSaved scripts/miner_1_20280323_explore_results.json')
