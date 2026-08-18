# -*- coding: utf-8 -*-
"""miner_1 2028-03-23 batch b3: fresh factor ideas.
Visible data through 2028-03-22. Gates: |IC| >= 0.007 and |ICIR| >= 0.084 at 10d
horizon on the 15-asset universe; post-gate max_abs_library_correlation < 0.5.
Focus areas NOT yet covered by library (which holds trend_r2, semi_down, mom 10/120d,
time_under_water, vol_of_vol, dxy/vix/wti beta, kurt, tail_ratio) or by b1/b2:
  - volume/liquidity (amihud, volume trend, volume-price corr)
  - vol regime change (rvol expansion 5/60)
  - vol asymmetry 60d, MA-distance vol-normalized
  - return autocorrelation (persistence)
  - momentum oscillators (MACD hist, MA crossover)
  - lottery/crash sensitivity (max/min daily ret over 20d)
  - drawdown recovery progress (recovery_ratio_60)
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
print('Volume non-null frac per asset:', {s: round(float(V[s].notna().mean()), 2) for s in C.columns})


def safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)


def build(name):
    # ---- volume / liquidity ----
    if name == 'amihud_illiq_20':
        # Amihud illiquidity: |ret| / volume, 20d mean. High = illiquid (premium?)
        illiq = safe_div(R.abs(), V)
        return illiq.rolling(20).mean()
    if name == 'vol_trend_20_60':
        # volume expansion: 5d avg volume / 60d avg volume (liquidity inflow)
        return V.rolling(5).mean() / V.rolling(60).mean().replace(0, np.nan)
    if name == 'vol_price_corr_20':
        # rolling corr between volume and daily return (participation/conviction)
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            x = pd.concat([V[s], R[s]], axis=1).dropna()
            out[s] = x.iloc[:, 0].rolling(20).corr(x.iloc[:, 1])
        return out
    # ---- vol regime ----
    if name == 'rvol_expansion_5_60':
        # short-term realized vol vs longer-term vol (regime shift)
        v5 = R.rolling(5).std()
        v60 = R.rolling(60).std()
        return safe_div(v5, v60)
    # ---- vol asymmetry (60d) ----
    if name == 'up_down_vol_60':
        up = R.where(R > 0)
        dn = R.where(R < 0)
        upv = up.rolling(60).std()
        dnv = dn.rolling(60).std()
        return safe_div(dnv, upv)
    # ---- MA distance, vol-normalized ----
    if name == 'ma_dist_vol_60':
        ma60 = C.rolling(60).mean()
        dist = safe_div(C, ma60) - 1.0
        return safe_div(dist, R.rolling(60).std())
    # ---- return autocorrelation (persistence) ----
    if name == 'ret_autocorr_10':
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            x = R[s]
            out[s] = x.rolling(10).apply(lambda w: w[:-1].corr(w[1:]) if len(w) > 2 else np.nan, raw=True)
        return out
    # ---- momentum oscillators ----
    if name == 'macd_hist_20':
        e12 = C.ewm(span=12, adjust=False).mean()
        e26 = C.ewm(span=26, adjust=False).mean()
        macd = e12 - e26
        sig = macd.ewm(span=9, adjust=False).mean()
        return safe_div(macd - sig, C)
    if name == 'crossover_ma_20_60':
        ma20 = C.rolling(20).mean()
        ma60 = C.rolling(60).mean()
        return safe_div(safe_div(ma20, ma60) - 1.0, R.rolling(20).std())
    # ---- lottery / crash sensitivity ----
    if name == 'max_ret_20':
        return R.rolling(20).max()
    if name == 'min_ret_20':
        return R.rolling(20).min()
    # ---- drawdown recovery progress ----
    if name == 'recovery_ratio_60':
        dd = C / C.cummax() - 1.0
        # 1 at trough (no recovery), ~0 after full recovery from 60d max drawdown
        min60 = dd.rolling(60).min()
        return safe_div(dd, min60)
    return None


CANDIDATES = ['amihud_illiq_20', 'vol_trend_20_60', 'vol_price_corr_20',
              'rvol_expansion_5_60', 'up_down_vol_60', 'ma_dist_vol_60',
              'ret_autocorr_10', 'macd_hist_20', 'crossover_ma_20_60',
              'max_ret_20', 'min_ret_20', 'recovery_ratio_60']

out = {'visible_through': str(C.index.max().date()), 'n_dates': len(C), 'n_assets': C.shape[1],
       'library_factors': L.LIB_FACTORS, 'results': {}}
for name in CANDIDATES:
    try:
        fp = build(name)
        if fp is None:
            print('\n[%s] build failed' % name)
            continue
        summ = L.full_validate(fp, R, horizon=10, label=name)
        ic, icir = summ['ic'], summ['icir']
        gate = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
        maxrho = summ['max_abs_library_correlation']
        rho_ok = maxrho < 0.5
        summ['pass_gate'] = bool(gate)
        summ['rho_ok'] = bool(rho_ok)
        out['results'][name] = {k: summ[k] for k in
                                ['label', 'horizon', 'ic', 'icir', 'ic_hit_ratio', 'n_ic_dates',
                                 'regime', 'coverage_asset_days', 'coverage_dates_ge8',
                                 'turnover_10d_rank', 'decay_ic_by_horizon',
                                 'max_abs_library_correlation', 'pass_gate', 'rho_ok']}
        print('%-22s ic=%+.4f icir=%+.4f n=%5d hit=%.3f cov=%.3f rho=%.3f gate=%s rho_ok=%s'
              % (name, ic, icir, summ['n_ic_dates'], summ['ic_hit_ratio'],
                 summ['coverage_dates_ge8'], maxrho, gate, rho_ok))
        print('    decay:', summ['decay_ic_by_horizon'])
        print('    regime:', {k: v['ic'] for k, v in summ.get('regime', {}).items()})
    except Exception as e:
        print('\n[%s] validation error: %s' % (name, e))
        out['results'][name] = {'error': str(e)}

with open('scripts/miner_1_20280323_explore_results_b3.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nSaved scripts/miner_1_20280323_explore_results_b3.json')
