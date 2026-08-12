# -*- coding: utf-8 -*-
"""miner_3 2028-02-10: explore novel factor batch (cycle data through 2028-02-09).
Gates: |IC| >= 0.007 and |ICIR| >= 0.084 at 10d horizon on the 15-asset universe.
Robustness sanity: require n_ic_dates >= 120 and coverage_dates_ge8 >= 0.6 for a
candidate to be considered a genuine gate pass (small-n spurious passes rejected).
Data quirks (verified): open missing ~27% except BTC/ETH; volume==0 for
SOX/XAU/COPPER/WTI/US10Y/CN10Y; high/low missing for SOX/US10Y/CN10Y -> close-only
factors preferred; open-based factors limited to recent window (treat small-n).
Novel angle this cycle: cross-sectional relative strength (mom vs universe median),
drawdown-recovery structure, price position within range, yield dynamics,
commodity-beta (WTI) for crash-reversal, skewness signed, momentum acceleration.
"""
import sys, json, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

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

def build(name):
    # ---- cross-sectional relative strength: 20d mom minus universe median 20d mom ----
    if name == 'rel_mom_20':
        return (C.pct_change(20).sub(C.pct_change(20).median(axis=1), axis=0))
    # ---- cross-sectional relative strength: 60d mom minus universe median 60d mom ----
    if name == 'rel_mom_60':
        return (C.pct_change(60).sub(C.pct_change(60).median(axis=1), axis=0))
    # ---- momentum acceleration: 20d mom minus 60d mom (short-term vs long-term) ----
    if name == 'mom_accel_20_60':
        return (C.pct_change(20) - C.pct_change(60)).replace([np.inf, -np.inf], np.nan)
    # ---- price position: close distance from 120d low normalized by 120d range ----
    if name == 'pos_120':
        lo = C.rolling(120).min()
        hi = C.rolling(120).max()
        return ((C - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)
    # ---- distance from 60d high (drawdown state) ----
    if name == 'dd_60':
        hi = C.rolling(60).max()
        return (C / hi - 1.0)
    # ---- drawdown recovery: fraction of 60d max drawdown recovered over last 20d ----
    if name == 'dd_recover_60x20':
        hi = C.rolling(60).max()
        dd_20 = C / hi - 1.0
        dd_0 = C.shift(20) / hi.shift(20) - 1.0
        return (dd_20 - dd_0).replace([np.inf, -np.inf], np.nan)
    # ---- oil beta 60d (beta to WTI returns) ----
    if name == 'wti_beta_60_alt':
        f = R['WTI']
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- signed skewness of 20d returns ----
    if name == 'skew_20_signed':
        return R.rolling(20).skew()
    # ---- yield momentum: 20d change in US10Y (higher = rates up = risk-off) ----
    if name == 'us10y_chg_20':
        return (C['US10Y'].diff(20) / C['US10Y'].shift(20)).to_frame('US10Y') \
            .reindex(columns=C.columns).fillna(0.0) if False else None
    # ---- cross-sectional vol rank: 20d vol minus universe median 20d vol ----
    if name == 'rel_vol_20':
        v = R.rolling(20).std()
        return (v.sub(v.median(axis=1), axis=0))
    # ---- up-day momentum: mean of positive-day returns over 20d (asymmetry) ----
    if name == 'upday_mean_20':
        pos = R.where(R > 0, np.nan)
        return pos.rolling(20).mean()
    # ---- down-day magnitude: mean of negative-day returns over 20d ----
    if name == 'downday_mean_20':
        neg = R.where(R < 0, np.nan)
        return neg.rolling(20).mean()
    # ---- cross-asset beta to equal-weight universe (60d) ----
    if name == 'uni_beta_60':
        f = R.mean(axis=1)
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- VIX level regime: VIX 20d mean vs its 120d mean (risk regime) ----
    if name == 'vix_level_20_120':
        return pd.DataFrame({s: (VIX.rolling(20).mean() / VIX.rolling(120).mean() - 1.0).values
                             for s in C.columns}, index=C.index)
    # ---- range contraction: 20d range / 120d range (compression) ----
    if name == 'range_ratio_20_120':
        r20 = C.rolling(20).max() - C.rolling(20).min()
        r120 = C.rolling(120).max() - C.rolling(120).min()
        return (r20 / r120).replace([np.inf, -np.inf], np.nan)
    # ---- high-low position: close relative to 20d high-low midpoint ----
    if name == 'hl_pos_20':
        hi = C.rolling(20).max()
        lo = C.rolling(20).min()
        return (2 * C - hi - lo) / (hi - lo).replace(0, np.nan)
    return None

CANDIDATES = ['rel_mom_20', 'rel_mom_60', 'mom_accel_20_60', 'pos_120', 'dd_60',
              'dd_recover_60x20', 'wti_beta_60_alt', 'skew_20_signed', 'rel_vol_20',
              'upday_mean_20', 'downday_mean_20', 'uni_beta_60', 'vix_level_20_120',
              'range_ratio_20_120', 'hl_pos_20']

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
    n_ok = summ['n_ic_dates'] >= 120
    cov_ok = summ['coverage_dates_ge8'] >= 0.6
    gate_ic = abs(summ['ic']) >= 0.007
    gate_icir = abs(summ['icir']) >= 0.084
    robust = n_ok and cov_ok
    results[name] = summ
    print('\n=== %s ===' % name)
    print('  IC=%.4f ICIR=%.4f hit=%.3f n=%d cov_asset=%.3f cov_dates_ge8=%.3f turn=%.3f'
          % (summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
             summ['coverage_asset_days'], summ['coverage_dates_ge8'], summ['turnover_10d_rank']))
    print('  decay:', summ['decay_ic_by_horizon'])
    print('  regime:', {k: v['ic'] for k, v in summ.get('regime', {}).items()})
    print('  max_abs_library_corr=%.3f' % summ['max_abs_library_correlation'])
    print('  GATE: ic=%s icir=%s robust(n>=120,cov>=0.6)=%s  => PASS=%s'
          % (gate_ic, gate_icir, robust, gate_ic and gate_icir and robust))

out = {'visible_through': str(C.index.max().date()), 'n_dates': int(len(C)),
       'n_assets': int(C.shape[1]), 'library_factors': lib_factors,
       'results': {k: {kk: vv for kk, vv in v.items() if kk != 'library_rho_by_factor'}
                   for k, v in results.items()}}
with open('scripts/miner_3_20280210_explore_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nSaved results to scripts/miner_3_20280210_explore_results.json')
