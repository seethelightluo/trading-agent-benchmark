#!/usr/bin/env python
"""miner_1 factor research batch T - 2030-07-29 (data through 2030-07-26)
1) Re-validate active library factors (drift check) through 2030-07-26.
2) Explore NEW candidate factors distinct from library + previously evicted/rejected.
Admission gates (15-asset cross-asset universe): |daily IC|>=0.0070, |daily ICIR|>=0.0840 at h=10.
"""
import pandas as pd, numpy as np, json
from scipy.stats import rankdata

CUR = pd.Timestamp('2030-07-26')  # last completed trading day before current date 2030-07-29
UNIV = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MIN_VALID = 8
H_ADMIT = 10
HORIZONS = [1,2,3,5,10,20]

def load_csv(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUR].set_index('date').sort_index()
    return df

closes, highs, lows, vols, opens = {}, {}, {}, {}, {}
for s in UNIV:
    df = load_csv(f'../persistent/stock_data/{s}.csv')
    closes[s] = df['close']; highs[s] = df['high']; lows[s] = df['low']; vols[s] = df['volume']; opens[s] = df['open']
close = pd.DataFrame(closes).ffill()
high  = pd.DataFrame(highs).ffill()
low   = pd.DataFrame(lows).ffill()
open_ = pd.DataFrame(opens).ffill()
volume = pd.DataFrame(vols)
idx = close.index
T = len(idx)
ret = close.pct_change()
print(f'panels: closes {close.shape} dates {idx[0].date()}..{idx[-1].date()} load ok', flush=True)

macro = {}
for m in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    macro[m] = load_csv(f'../persistent/index_data/{m}.csv')['close'].reindex(idx).ffill()

FR = {h: (close.shift(-h)/close - 1.0).values for h in HORIZONS}
FR_DF = {h: close.shift(-h)/close - 1.0 for h in HORIZONS}

# ---------- active library factor builders ----------
def f_vol_adj_mom_accel():
    mom20 = close/close.shift(20) - 1; mom60 = close/close.shift(60) - 1
    return (mom20 - mom60) / ret.rolling(20).std()

def f_dn_mkt_beta():
    mkt = ret.mean(axis=1); dn = mkt.where(mkt < 0)
    return ret.rolling(60).cov(dn) / dn.rolling(60).var()

def f_rate_beta_cn10y():
    cn = close['CN10Y'].pct_change()
    return ret.rolling(60).cov(cn) / cn.rolling(60).var()

# ---------- new candidate factor builders (batch T) ----------
def f_dxy_beta_60d():
    dxy = macro['DXY'].pct_change()
    return ret.rolling(60).cov(dxy) / dxy.rolling(60).var()

def f_us10y_beta_60d():
    u10 = close['US10Y'].pct_change()
    return ret.rolling(60).cov(u10) / u10.rolling(60).var()

def f_vix_beta_60d():
    vix = macro['VIX'].pct_change()
    return ret.rolling(60).cov(vix) / vix.rolling(60).var()

def f_eff_ratio_60():
    return (close/close.shift(60)-1).abs() / ret.abs().rolling(60).sum()

def f_skew_60():
    return ret.rolling(60).skew()

def f_drawdown_60():
    return close/close.rolling(60, min_periods=20).max() - 1.0

def f_range_pos_20():
    return ((close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min())).rolling(20).mean()

def f_overnight_mom_20():
    gap = open_/close.shift(1) - 1.0
    return gap.rolling(20).mean()

def f_bond_stock_corr_60():
    u10 = close['US10Y'].pct_change()
    return ret.rolling(60).corr(u10)

def f_crypto_beta_60d():
    btc = close['BTC'].pct_change()
    return ret.rolling(60).cov(btc) / btc.rolling(60).var()

def f_high_low_corr_20():
    # correlation of asset returns with cross-sectional high-minus-low spread (dispersion regime)
    disp = (high/low - 1.0).mean(axis=1)
    return ret.rolling(20).corr(disp)

# ---------- evaluation ----------
def evaluate(name, fac_np, exp_sign=1):
    ics = {}
    for h in HORIZONS:
        fr = FR[h]
        rows = []
        for t in range(T):
            fv = fac_np[t]; frv = fr[t]
            m = ~(np.isnan(fv) | np.isnan(frv))
            if m.sum() >= MIN_VALID:
                rows.append((t, float(np.corrcoef(fv[m], frv[m])[0,1])))
        ics[h] = np.array([r[1] for r in rows])
    ic10 = ics[H_ADMIT]
    valid = ~np.isnan(ic10)
    n = int(valid.sum())
    ic = float(np.nanmean(ic10))
    icir = float(np.nanmean(ic10)/np.nanstd(ic10)) if np.nanstd(ic10) > 0 else 0.0
    hit = float(np.mean(np.sign(ic10[valid]) == np.sign(ic)))
    ic_std = float(np.nanstd(ic10))
    fac_np2 = fac_np
    cov_ad = float(np.mean(np.sum(~np.isnan(fac_np2), axis=1) >= MIN_VALID))
    cov_d8 = float(np.mean(np.sum(~np.isnan(fac_np2), axis=1) >= 8))
    trs = []
    for i in range(T - 10):
        a = fac_np2[i]; b = fac_np2[i+10]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() >= MIN_VALID:
            trs.append(float(np.abs(rankdata(a[m]) - rankdata(b[m])).mean()))
    turn = float(np.mean(trs)) if trs else np.nan
    lib = {'vol_adj_mom_accel_20x60': f_vol_adj_mom_accel().values,
           'dn_mkt_beta_60d': f_dn_mkt_beta().values,
           'rate_beta_cn10y_60d': f_rate_beta_cn10y().values}
    best_rho, best_f = 0.0, None
    for lf, lp in lib.items():
        m = ~(np.isnan(fac_np) | np.isnan(lp))
        if m.sum() < 100:
            continue
        rf = rankdata(fac_np[m]); rg = rankdata(lp[m])
        rf = rf - rf.mean(); rg = rg - rg.mean()
        denom = np.sqrt((rf*rf).sum()*(rg*rg).sum())
        rho = (rf*rg).sum()/denom if denom > 0 else 0.0
        if abs(rho) > abs(best_rho):
            best_rho, best_f = rho, lf
    res = dict(name=name, exp_sign=exp_sign, ic=ic, icir=icir, ic_hit=hit, n_dates=n,
               ic_std=ic_std, cov_ad=cov_ad, cov_d8=cov_d8, turn=turn,
               max_rho=float(best_rho), max_rho_f=best_f,
               decay={str(h): round(float(np.nanmean(ics[h])), 4) for h in HORIZONS})
    for win in [250, 500, 750]:
        if n >= win:
            v = ic10[valid][-win:]
            res[f'ic_r{win}'] = float(v.mean())
            res[f'icir_r{win}'] = float(v.mean()/v.std()) if v.std() > 0 else 0.0
    return res

print('\n=== ACTIVE LIBRARY RE-VALIDATION (through 2030-07-26) ===', flush=True)
for name, fn in [('vol_adj_mom_accel_20x60', f_vol_adj_mom_accel),
                 ('dn_mkt_beta_60d', f_dn_mkt_beta),
                 ('rate_beta_cn10y_60d', f_rate_beta_cn10y)]:
    r = evaluate(name, fn().values, 1)
    print(f"[active] {name}: IC={r['ic']:.4f} ICIR={r['icir']:.4f} hit={r['ic_hit']:.3f} n={r['n_dates']} cov_d8={r['cov_d8']:.3f} r250={r.get('ic_r250',np.nan):.4f}/{r.get('icir_r250',np.nan):.3f} decay10={r['decay']['10']:.4f}", flush=True)

print('\n=== CANDIDATE FACTORS (h=10 admission) ===', flush=True)
CANDIDATES = [
    ('dxy_beta_60d', 'beta to DXY (USD sensitivity)', f_dxy_beta_60d, 1),
    ('us10y_beta_60d', 'beta to US10Y yield changes', f_us10y_beta_60d, 1),
    ('vix_beta_60d', 'beta to VIX changes (risk sentiment)', f_vix_beta_60d, -1),
    ('eff_ratio_60', 'Kaufman efficiency ratio 60d (trend quality)', f_eff_ratio_60, 1),
    ('skew_60', '60d return skewness (tail risk)', f_skew_60, -1),
    ('drawdown_60', 'distance from 60d high (dip level)', f_drawdown_60, -1),
    ('range_pos_20', 'avg close position in 20d range', f_range_pos_20, 1),
    ('overnight_mom_20', 'avg overnight gap 20d', f_overnight_mom_20, 1),
    ('bond_stock_corr_60', 'corr with US10Y returns 60d', f_bond_stock_corr_60, 1),
    ('crypto_beta_60d', 'beta to BTC returns 60d', f_crypto_beta_60d, 1),
    ('hl_disp_corr_20', 'corr with cross-sectional dispersion', f_high_low_corr_20, 1),
]
results = []
for name, desc, fn, esign in CANDIDATES:
    fac = fn().values
    r = evaluate(name, fac, esign)
    results.append((name, desc, esign, r))
    flag = 'PASS' if (abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084 and r['n_dates'] >= 200) else 'fail'
    print(f"[{flag}] {name} | IC={r['ic']:.4f} ICIR={r['icir']:.4f} hit={r['ic_hit']:.3f} n={r['n_dates']} cov_ad={r['cov_ad']:.3f} cov_d8={r['cov_d8']:.3f} turn={r['turn']:.2f} |rho_lib|={abs(r['max_rho']):.3f}({r['max_rho_f']}) r250={r.get('ic_r250',np.nan):.4f}/{r.get('icir_r250',np.nan):.3f} decay={r['decay']}", flush=True)

print('\n=== SUMMARY TABLE ===', flush=True)
print(f"{'factor':24s} {'IC':>8s} {'ICIR':>8s} {'hit':>6s} {'n':>5s} {'covD8':>6s} {'turn':>6s} {'rhoLib':>7s} {'status':>6s}")
for name, desc, esign, r in results:
    status = 'PASS' if (abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084 and r['n_dates'] >= 200) else ''
    print(f"{name:24s} {r['ic']:8.4f} {r['icir']:8.4f} {r['ic_hit']:6.3f} {r['n_dates']:5d} {r['cov_d8']:6.3f} {r['turn']:6.2f} {abs(r['max_rho']):7.3f} {status:6s}")

import pickle
with open('scripts/_miner1_20300729_results.pkl','wb') as fh:
    pickle.dump({'results': [(n,d,e,r) for n,d,e,r in results]}, fh)
print('saved results pkl', flush=True)
