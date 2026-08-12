# -*- coding: utf-8 -*-
"""miner_3 2028-03-23: explore novel factor batch (data through 2028-03-22).
Gates: |IC| >= 0.007 and |ICIR| >= 0.084 at 10d horizon on the 15-asset universe.
Robustness sanity: n_ic_dates >= 120 and coverage_dates_ge8 >= 0.6.
Novel angles this cycle (not in library / not previously evicted):
  - Kaufman efficiency ratio (trend quality, different math than trend_r2)
  - extremes asymmetry (max gain / max loss), up/down mean ratio
  - vol term structure (5d/60d, 10d/60d)
  - cross-asset systematic-ness (mean pairwise corr), downside beta
  - China-market beta (000300.SH), rate beta (US10Y), JPY beta (USDJPY), DXY-up-day beta
  - risk-adjusted relative momentum, 20d new-high proximity
  - 60d skew, drawdown-speed 60x10, EWMA momentum
  - NEW: 5d serial correlation (mean-reversion proxy), volume-trend-weighted momentum,
    conditional momentum (trend filter), US10Y-CN10Y yield spread change,
    crypto beta (BTC), 120d max drawdown, cross-sectional return dispersion tilt
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


def rolling_beta_cond(x, f, win, cond):
    """Beta computed only on days where cond (boolean series) is True."""
    xc = x.where(cond)
    fc = f.where(cond)
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        xv = xc[s]
        m = xv.notna() & fc.notna()
        out[s] = xv.rolling(win, min_periods=15).cov(fc) / fc.rolling(win, min_periods=15).var()
    return out.replace([np.inf, -np.inf], np.nan)


def build(name):
    # ---- 1. Kaufman efficiency ratio signed by 20d direction ----
    if name == 'eff_ratio_20_signed':
        path = R.abs().rolling(20).sum()
        net = (C - C.shift(20)).abs()
        eff = (net / path).replace([np.inf, -np.inf], np.nan)
        return eff * np.sign(C.pct_change(20))
    # ---- 2. extremes asymmetry: max 20d gain / |max 20d loss| ----
    if name == 'max_gain_loss_20':
        g = R.rolling(20).max()
        l = R.rolling(20).min()
        return (g / l.abs()).replace([np.inf, -np.inf], np.nan)
    # ---- 3. up/down mean ratio over 60d ----
    if name == 'updown_ratio_60':
        pos = R.where(R > 0, np.nan).rolling(60).mean()
        neg = R.where(R < 0, np.nan).rolling(60).mean()
        return (pos / neg.abs()).replace([np.inf, -np.inf], np.nan)
    # ---- 4. vol term structure: 5d vol / 60d vol ----
    if name == 'vol_ts_5_60':
        v5 = R.rolling(5).std()
        v60 = R.rolling(60).std()
        return (v5 / v60).replace([np.inf, -np.inf], np.nan)
    # ---- 5. vol term structure: 10d vol / 60d vol ----
    if name == 'vol_ts_10_60':
        v10 = R.rolling(10).std()
        v60 = R.rolling(60).std()
        return (v10 / v60).replace([np.inf, -np.inf], np.nan)
    # ---- 6. systematic-ness: mean pairwise correlation with other assets (60d) ----
    if name == 'uni_corr_60':
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for d in R.index:
            w = R.loc[:d].tail(60)
            if len(w) < 30:
                continue
            c = w.corr()
            for s in R.columns:
                others = [o for o in R.columns if o != s]
                out.loc[d, s] = c.loc[s, others].mean()
        return out
    # ---- 7. downside beta: beta to equal-weight universe on down days (60d) ----
    if name == 'downside_beta_60':
        f = R.mean(axis=1)
        return rolling_beta_cond(R, f, 60, f < 0)
    # ---- 8. China market beta: beta to 000300.SH returns (60d) ----
    if name == 'chn_beta_60':
        f = R['000300.SH']
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- 9. rate beta: beta to US10Y yield changes (60d) ----
    if name == 'rate_beta_60':
        f = R['US10Y']
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- 10. JPY beta: beta to USDJPY returns (60d) ----
    if name == 'jpy_beta_60':
        fj = USDJPY.pct_change()
        return pd.DataFrame({s: rolling_beta(R[s], fj, 60) for s in R.columns}, index=R.index)
    # ---- 11. DXY-up-day beta (60d) ----
    if name == 'dxy_up_beta_60':
        fd = DXY.pct_change()
        return rolling_beta_cond(R, fd, 60, fd > 0)
    # ---- 12. risk-adjusted relative momentum: (20d ret - median) / 20d vol ----
    if name == 'rel_mom_vol_20':
        m = C.pct_change(20).sub(C.pct_change(20).median(axis=1), axis=0)
        v = R.rolling(20).std()
        return (m / v).replace([np.inf, -np.inf], np.nan)
    # ---- 13. 20d new-high proximity: close / rolling_max(20) - 1 ----
    if name == 'hi_prox_20':
        hi = C.rolling(20).max()
        return (C / hi - 1.0).replace([np.inf, -np.inf], np.nan)
    # ---- 14. 60d signed skewness ----
    if name == 'skew_60_signed':
        return R.rolling(60).skew()
    # ---- 15. drawdown-speed: 10d change in (C/high60 - 1) ----
    if name == 'dd_speed_60x10':
        hi = C.rolling(60).max()
        dd = C / hi - 1.0
        return (dd - dd.shift(10)).replace([np.inf, -np.inf], np.nan)
    # ---- 16. EWMA momentum (20d half-life ~5d) ----
    if name == 'ewma_mom_20':
        w = np.exp(-np.log(2) * np.arange(1, 21) / 5.0)
        w = w / w.sum()
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            out[s] = R[s].rolling(20).apply(lambda x: np.dot(x, w), raw=True)
        return out
    # ---- 17. 5d serial correlation of returns (mean-reversion proxy, negated) ----
    if name == 'ser_corr_5':
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            out[s] = R[s].rolling(20).apply(
                lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 8 else np.nan, raw=False)
        return out.replace([np.inf, -np.inf], np.nan)
    # ---- 18. volume-trend-weighted momentum: 20d mom * (20d vol trend) ----
    if name == 'volwt_mom_20':
        vt = V.rolling(20).mean() / V.rolling(60).mean()
        return (C.pct_change(20) * vt).replace([np.inf, -np.inf], np.nan)
    # ---- 19. conditional momentum: 20d mom only when close > MA60, else 0 ----
    if name == 'cond_mom_20_60':
        m = C.pct_change(20)
        ma60 = C.rolling(60).mean()
        return m.where(C > ma60, 0.0)
    # ---- 20. yield spread change: (US10Y - CN10Y) 20d change ----
    if name == 'yld_spread_chg_20':
        sp = C['US10Y'] - C['CN10Y']
        chg = sp.diff(20)
        return pd.DataFrame({s: chg.values for s in C.columns}, index=C.index)
    # ---- 21. crypto beta: beta to BTC returns (60d) ----
    if name == 'crypto_beta_60':
        f = R['BTC']
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    # ---- 22. 120d max drawdown (depth, negated so deeper = lower) ----
    if name == 'max_dd_120':
        hi = C.rolling(120).max()
        dd = C / hi - 1.0
        return dd.rolling(120).min()
    # ---- 23. cross-sectional dispersion tilt: asset's |mom20 - median| ----
    if name == 'disp_tilt_20':
        m = C.pct_change(20)
        med = m.median(axis=1)
        return (m - med).abs()
    return None


CANDIDATES = ['eff_ratio_20_signed', 'max_gain_loss_20', 'updown_ratio_60',
              'vol_ts_5_60', 'vol_ts_10_60', 'uni_corr_60', 'downside_beta_60',
              'chn_beta_60', 'rate_beta_60', 'jpy_beta_60', 'dxy_up_beta_60',
              'rel_mom_vol_20', 'hi_prox_20', 'skew_60_signed', 'dd_speed_60x10',
              'ewma_mom_20', 'ser_corr_5', 'volwt_mom_20', 'cond_mom_20_60',
              'yld_spread_chg_20', 'crypto_beta_60', 'max_dd_120', 'disp_tilt_20']

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
with open('scripts/miner_3_20280323_explore_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nSaved results to scripts/miner_3_20280323_explore_results.json')
