# -*- coding: utf-8 -*-
"""miner_3 2029-03-08: exploration batch (cross-asset beta / reversal / volume / macro-conditional).
Gates: |IC|>=0.0070, |ICIR|>=0.0840 at 10d horizon. Validation window: full panel 2020-01-01..2029-03-07,
plus recent-regime subperiods (2027+, 2028+, 2029+) reported for robustness.
No persistence in this script; only diagnostics."""
import sys, json, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

# extend library list with all effective factors on disk for rho audit
lib_all = []
for p in sorted(os.listdir('factors')):
    if p.endswith('.json') and p != 'factor_ensemble.json':
        try:
            d = json.load(open('factors/' + p))
            if d.get('validation', {}).get('status') == 'EFFECTIVE':
                lib_all.append(d['factor_id'])
        except Exception:
            pass
L.LIB_FACTORS = lib_all
print('Library factors for rho audit (%d): %s' % (len(lib_all), lib_all))

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))

def load_macro(name):
    df = pd.read_csv('../persistent/index_data/%s.csv' % name, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

DXY = load_macro('DXY'); USDJPY = load_macro('USDJPY')
EURUSD = load_macro('EURUSD'); USDCNY = load_macro('USDCNY'); VIX = load_macro('VIX')
RDXY = DXY.pct_change(); RVIX = VIX.pct_change(); RCNY = USDCNY.pct_change()
print('Macros loaded through %s' % DXY.index.max().date())

def roll_beta(x, ref, win=60, minp=60):
    cov = x.rolling(win, min_periods=minp).cov(ref)
    var = ref.rolling(win, min_periods=minp).var()
    return cov.div(var, axis=0).replace([np.inf, -np.inf], np.nan)

def roll_corr(x, ref, win=60, minp=60):
    return x.rolling(win, min_periods=minp).corr(ref).replace([np.inf, -np.inf], np.nan)

# ---------------- candidate factor panels ----------------
factors = {}

# 1) xau_beta_60 : 60d beta of asset vs XAU (safe-haven sensitivity)
factors['xau_beta_60'] = roll_beta(R, R['XAU'], 60)
# 2) copper_beta_60 : 60d beta vs COPPER (cyclical sensitivity)
factors['copper_beta_60'] = roll_beta(R, R['COPPER'], 60)
# 3) bond_beta_60 : 60d beta vs US10Y returns (bond-price-like sensitivity)
factors['bond_beta_60'] = roll_beta(R, R['US10Y'], 60)
# 4) spx_beta_60 : 60d beta vs SPX (market beta)
factors['spx_beta_60'] = roll_beta(R, R['SPX'], 60)
# 5) safe_haven_60 : corr(XAU)+corr(US10Y)-corr(SPX) over 60d (defensive composite)
factors['safe_haven_60'] = (roll_corr(R, R['XAU'], 60) + roll_corr(R, R['US10Y'], 60) - roll_corr(R, R['SPX'], 60))
# 6) rev_5d_skip1 : short-term reversal, -5d return skipping 1 day
factors['rev_5d_skip1'] = -(C.shift(1) / C.shift(6) - 1.0)
# 7) stoch_20 : stochastic oscillator position (C - min20)/(max20 - min20)
rng = H.rolling(20).max() - Lw.rolling(20).min()
factors['stoch_20'] = (C - Lw.rolling(20).min()) / rng.replace(0, np.nan)
# 8) down_up_20 : downside capture ratio |mean(neg)|/mean(pos) over 20d
negm = -R.where(R < 0, 0.0).rolling(20).mean()
posm = R.where(R > 0, 0.0).rolling(20).mean()
factors['down_up_20'] = negm / posm.replace(0, np.nan)
# 9) dist_low_20 : distance above 20d low (oversold/rebound indicator)
factors['dist_low_20'] = C / Lw.rolling(20).min() - 1.0
# 10) vwap_dev_20 : deviation of close from 20d volume-weighted typical price
tp = (H + Lw + C) / 3.0
vwap20 = (tp * V).rolling(20).sum() / V.rolling(20).sum().replace(0, np.nan)
factors['vwap_dev_20'] = C / vwap20 - 1.0
# 11) vol_z_20 : 20d volume z-score (flow surge)
factors['vol_z_20'] = (V - V.rolling(20).mean()) / V.rolling(20).std().replace(0, np.nan)
# 12) dxy_beta_up_60 : 60d beta vs DXY returns conditional on DXY up days
dxy_up = RDXY.where(RDXY > 0)
factors['dxy_beta_up_60'] = roll_beta(R, dxy_up, 60, minp=30)

# ---- new 2029-03-08 candidates: whipsaw/bearish regime focus ----
# 13) vol_scaled_rev_5 : 5d reversal normalized by 20d vol (z-score of short return)
r5 = C.shift(1) / C.shift(6) - 1.0
vol20 = R.rolling(20).std()
factors['vol_scaled_rev_5'] = -(r5 / vol20.replace(0, np.nan))
# 14) range_pos_20 : (C - L20)/(H20 - L20) intraday-range position (0=low,1=high)
factors['range_pos_20'] = (C - Lw.rolling(20).min()) / (H.rolling(20).max() - Lw.rolling(20).min()).replace(0, np.nan)
# 15) streak_10 : consecutive same-sign run length (up=+, down=-) over last 10d
def streak(s):
    out = np.zeros(len(s))
    run = 0
    for i in range(len(s)):
        r = s.iloc[i]
        if pd.notna(r):
            if r > 0:
                run = run + 1 if run > 0 else 1
            elif r < 0:
                run = run - 1 if run < 0 else -1
            else:
                run = 0
        out[i] = run
    return pd.Series(out, index=s.index)
factors['streak_10'] = R.apply(streak).clip(-10, 10)
# 16) rel_mom_20_vs_spx : 20d excess return vs SPX (cross-sectional relative momentum)
factors['rel_mom_20_vs_spx'] = (C / C.shift(20) - 1.0).sub((C['SPX'] / C['SPX'].shift(20) - 1.0), axis=0)
# 17) vix_cond_rev_5 : 5d reversal active only when VIX above its 60d rolling median (stress mean-reversion)
vix_hi = (VIX > VIX.rolling(60).median()).astype(float)
factors['vix_cond_rev_5'] = -(C.shift(1) / C.shift(6) - 1.0).mul(vix_hi, axis=0)
# 18) usdcny_beta_60 : 60d beta vs USDCNY returns (China FX stress; relevant for CN/HSI sleeves)
factors['usdcny_beta_60'] = roll_beta(R, RCNY, 60, minp=40)
# 19) usd_jpy_mom_60 : 60d USDJPY carry-trend macro signal (same value across all assets)
factors['usd_jpy_mom_60'] = pd.DataFrame({c: (USDJPY / USDJPY.shift(60) - 1.0) for c in C.columns}, index=C.index)
# 20) ret_vol_ratio_10 : 10d return / 10d vol (risk-adjusted short momentum)
r10 = C.shift(1) / C.shift(11) - 1.0
factors['ret_vol_ratio_10'] = r10 / R.rolling(10).std().replace(0, np.nan)

# ---------------- validation ----------------
def recent_summary(s):
    out = {}
    for name, lo in [("2027+", "2027-01-01"), ("2028+", "2028-01-01"), ("2029", "2029-01-01")]:
        sub = s[s.index >= lo]
        if len(sub) >= 20:
            out[name] = {'ic': round(sub.mean(), 4),
                         'icir': round(sub.mean() / sub.std(), 4) if sub.std() > 0 else 0.0,
                         'n': int(len(sub))}
    return out

results = {}
for fid, panel in factors.items():
    s = L.rank_ic(panel, R.shift(-10))
    if s is None or len(s) < 20:
        results[fid] = {'error': 'insufficient IC dates'}
        print('%s | ERROR: insufficient IC dates' % fid)
        continue
    summ = L.summarize(s, 10, fid)
    summ['regime_recent'] = recent_summary(s)
    summ['decay_ic_by_horizon'] = L.decay_analysis(panel, R)
    summ['coverage'] = L.coverage_turnover(panel, R, 10)
    rhos, maxrho = L.library_max_rho(panel)
    summ['max_abs_library_correlation'] = maxrho
    results[fid] = summ
    gate = (abs(summ['ic']) >= 0.0070) and (abs(summ['icir']) >= 0.0840)
    print('%s | ic=%.4f icir=%.4f hit=%.3f n=%d cov_ge8=%.2f maxrho=%.2f | %s' % (
        fid, summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
        summ['coverage']['coverage_dates_ge8'], maxrho, 'PASS' if gate else 'fail'))

with open('scripts/miner_3_20290308_explore_results.json', 'w') as f:
    json.dump({'visible_through': str(C.index.max().date()), 'n_dates': len(C), 'n_assets': C.shape[1],
               'library_factors': lib_all, 'results': results}, f, indent=1, default=str)
print('\nSaved scripts/miner_3_20290308_explore_results.json')
