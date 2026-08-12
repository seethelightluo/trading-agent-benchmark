# -*- coding: utf-8 -*-
"""miner_3 2028-01-27: explore novel factor batch.
Visible data through previous completed trading day 2028-01-26.
Gates: |IC| >= 0.007 and |ICIR| >= 0.084 at 10d horizon on the 15-asset universe;
post-gate constraint: max_abs_library_correlation < 0.5 (pairwise, artifacts).
Data quirks (verified): open missing ~27% for all assets except BTC/ETH;
volume == 0 for SOX/XAU/COPPER/WTI/US10Y/CN10Y -> no volume factors;
high/low missing for SOX/US10Y/CN10Y -> close-only factors preferred.
Novel angle this cycle: gap/overnight structure (open-based), run-streak /
win-consistency, cross-asset betas (NDX/COPPER/USDCNY/USDJPY/yield-spread/VIX),
robust location (median), short-window momentum contrast, conditional momentum.
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
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))

def load_macro(name):
    df = pd.read_csv('../persistent/index_data/%s.csv' % name, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

DXY = load_macro('DXY')
USDCNY = load_macro('USDCNY')
USDJPY = load_macro('USDJPY')
VIX = load_macro('VIX')

def rolling_beta(x, f, win):
    cov = x.rolling(win).cov(f)
    var = f.rolling(win).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)

def max_streak_pos(s, win):
    """rolling max length of consecutive positive-day runs over win days"""
    u = (s > 0).astype(int)
    g = (u != u.shift()).cumsum()
    run = u * (g.groupby(g).cumcount() + 1)
    return run.rolling(win).max()

def build(name):
    # ---- upside consistency: fraction of positive days over 20d ----
    if name == 'win_rate_20':
        return (R > 0).rolling(20).mean()
    # ---- momentum run strength: max consecutive up days over 20d ----
    if name == 'streak_max_20':
        return pd.DataFrame({s: max_streak_pos(R[s], 20) for s in R.columns}, index=R.index)
    # ---- systematic gap direction: mean overnight gap (open/prev_close-1) over 20d ----
    if name == 'gap_avg_20':
        gap = O / C.shift(1) - 1.0
        return gap.rolling(20).mean()
    # ---- gap instability: std of overnight gaps over 20d ----
    if name == 'gap_vol_20':
        gap = O / C.shift(1) - 1.0
        return gap.rolling(20).std()
    # ---- overnight share of total move (open-based) over 20d ----
    if name == 'overnight_share_20':
        gap = (O / C.shift(1) - 1.0).abs()
        intra = (C / O - 1.0).abs()
        num = gap.rolling(20).sum()
        den = (gap + intra).rolling(20).sum()
        return (num / den).replace([np.inf, -np.inf], np.nan)
    # ---- candle body size relative to realized vol over 20d ----
    if name == 'body_ratio_20':
        body = (C - O).abs() / C
        return (body.rolling(20).mean() / R.rolling(20).std()).replace([np.inf, -np.inf], np.nan)
    # ---- tech beta 60d (beta to NDX returns) ----
    if name == 'ndx_beta_60':
        f = R['NDX']
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- industrial beta 60d (beta to COPPER returns) ----
    if name == 'copper_beta_60':
        f = R['COPPER']
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- China FX beta 60d (beta to USDCNY returns) ----
    if name == 'usdcny_beta_60':
        f = USDCNY.pct_change()
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- carry beta 60d (beta to USDJPY returns) ----
    if name == 'usdjpy_beta_60':
        f = USDJPY.pct_change()
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- yield-curve spread beta 60d (beta to d(CN10Y - US10Y)) ----
    if name == 'yield_gap_beta_60':
        spread = C['CN10Y'] - C['US10Y']
        f = spread.diff()
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- unconditional VIX beta 60d (beta to VIX pct change) ----
    if name == 'vix_beta_60':
        f = VIX.pct_change()
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- short-window market correlation 20d (corr with SPX returns) ----
    if name == 'spx_corr_20':
        f = R['SPX']
        return pd.DataFrame({s: R[s].rolling(20).corr(f) for s in R.columns}, index=R.index)
    # ---- robust location: median daily return over 20d ----
    if name == 'med_ret_20':
        return R.rolling(20).median()
    # ---- short momentum contrast: last 5d vs prior 15d ----
    if name == 'mom_5_15':
        last5 = C.pct_change(5)
        prev15 = C.pct_change(20) - C.pct_change(5)
        return (last5 - prev15).replace([np.inf, -np.inf], np.nan)
    # ---- up-gap ratio over 20d (open-based) ----
    if name == 'up_gap_ratio_20':
        gap = O / C.shift(1) - 1.0
        return (gap > 0).rolling(20).mean()
    # ---- conditional momentum: 20d mom only when VIX above 60d MA ----
    if name == 'mom_20_high_vix':
        vix_hi = (VIX > VIX.rolling(60).mean()).astype(float)
        return (C.pct_change(20) * vix_hi.values).replace([np.inf, -np.inf], np.nan)
    # ---- return per unit of 20d close range ----
    if name == 'ret_range_20':
        rng = C.rolling(20).max() - C.rolling(20).min()
        return (C.pct_change(20) / rng).replace([np.inf, -np.inf], np.nan)
    return None

CANDIDATES = ['win_rate_20', 'streak_max_20', 'gap_avg_20', 'gap_vol_20',
              'overnight_share_20', 'body_ratio_20', 'ndx_beta_60', 'copper_beta_60',
              'usdcny_beta_60', 'usdjpy_beta_60', 'yield_gap_beta_60', 'vix_beta_60',
              'spx_corr_20', 'med_ret_20', 'mom_5_15', 'up_gap_ratio_20',
              'mom_20_high_vix', 'ret_range_20']

results = {}
for name in CANDIDATES:
    fp = build(name)
    if fp is None:
        print('\n[%s] build failed' % name)
        continue
    try:
        summ = L.full_validate(fp, R, horizon=10, label=name)
    except Exception as e:
        print('\n[%s] validation error: %s' % (name, e))
        continue
    gate_ic = abs(summ['ic']) >= 0.007
    gate_icir = abs(summ['icir']) >= 0.084
    results[name] = summ
    print('\n=== %s ===' % name)
    print('  IC=%.4f ICIR=%.4f hit=%.3f n=%d cov_asset=%.3f cov_dates_ge8=%.3f turn=%.3f'
          % (summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
             summ['coverage_asset_days'], summ['coverage_dates_ge8'], summ['turnover_10d_rank']))
    print('  decay:', summ['decay_ic_by_horizon'])
    print('  regime:', {k: v['ic'] for k, v in summ.get('regime', {}).items()})
    print('  max_abs_library_corr=%.3f  rho_by_fac=%s' % (summ['max_abs_library_correlation'],
          {k: v for k, v in summ.get('library_rho_by_factor', {}).items() if v is not None}))
    print('  GATE PASS: %s (ic=%s icir=%s)' % (gate_ic and gate_icir, gate_ic, gate_icir))

out = {'visible_through': str(C.index.max().date()), 'n_dates': int(len(C)),
       'n_assets': int(C.shape[1]), 'library_factors': lib_factors,
       'results': {k: {kk: vv for kk, vv in v.items() if kk != 'library_rho_by_factor'}
                   for k, v in results.items()}}
with open('scripts/miner_3_20280127_explore_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nSaved results to scripts/miner_3_20280127_explore_results.json')
