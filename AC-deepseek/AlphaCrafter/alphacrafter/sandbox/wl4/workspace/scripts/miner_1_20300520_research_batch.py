#!/usr/bin/env python
"""miner_1 factor research batch v2 (vectorized) - 2030-05-20
1) Re-validate active library factors (drift check) through current visible date.
2) Explore candidate factors (vol/quality, liquidity, macro-beta, reversal).
Admission gates (15-asset cross-asset universe): |daily IC|>=0.0070, |daily ICIR|>=0.0840 at h=10.
"""
import pandas as pd, numpy as np, glob, os, json, hashlib, zlib, base64
from scipy.stats import rankdata

CUR = pd.Timestamp('2030-05-20')
UNIV = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MIN_VALID = 8
H_ADMIT = 10
HORIZONS = [1,2,3,5,10,20]

def load_csv(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUR].set_index('date').sort_index()
    return df

closes, highs, lows, vols = {}, {}, {}, {}
for s in UNIV:
    df = load_csv(f'../persistent/stock_data/{s}.csv')
    closes[s] = df['close']; highs[s] = df['high']; lows[s] = df['low']; vols[s] = df['volume']
close = pd.DataFrame(closes).ffill()
high  = pd.DataFrame(highs).ffill()
low   = pd.DataFrame(lows).ffill()
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

# ---------- factor builders (return numpy panels) ----------
def panel_to_np(fn):
    p = fn()
    return p.values

def f_vol_adj_mom_accel():
    mom20 = close/close.shift(20) - 1; mom60 = close/close.shift(60) - 1
    return (mom20 - mom60) / ret.rolling(20).std()

def f_dn_mkt_beta():
    mkt = ret.mean(axis=1); dn = mkt.where(mkt < 0)
    return ret.rolling(60).cov(dn) / dn.rolling(60).var()

def f_rate_beta_cn10y():
    cn = close['CN10Y'].pct_change()
    return ret.rolling(60).cov(cn) / cn.rolling(60).var()

def f_parkinson_vol_20():
    hl = np.log(high/low)
    return np.sqrt((0.5*hl**2).rolling(20).mean())

def f_range_ratio_20():
    return ((high-low)/close).rolling(20).mean()

def f_max_dd_60():
    return close/close.rolling(60, min_periods=20).max() - 1.0

def f_downside_vol_20():
    return ret.where(ret < 0).rolling(20).std()

def f_efficiency_ratio_20():
    return (close/close.shift(20)-1).abs() / ret.abs().rolling(20).sum()

def f_stochastic_pos_20():
    return (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min())

def f_win_rate_20():
    return (ret > 0).rolling(20).mean()

def f_avg_gain_loss_20():
    pos = ret.where(ret>0).rolling(20).mean(); neg = ret.where(ret<0).rolling(20).mean().abs()
    return pos/neg

def f_volume_trend_10_60():
    return volume.rolling(10).mean() / volume.rolling(60).mean()

def f_amihud_20():
    return (ret.abs()/volume.replace(0,np.nan)).rolling(20).mean()

def make_beta(sig, win=60):
    return ret.rolling(win).cov(sig) / sig.rolling(win).var()

def f_dxy_beta(): return make_beta(macro['DXY'].pct_change())
def f_vix_beta(): return make_beta(macro['VIX'].pct_change())
def f_wti_beta(): return make_beta(ret['WTI'])
def f_xau_beta(): return make_beta(ret['XAU'])
def f_corr_us10y_60():
    u = close['US10Y'].pct_change()
    return ret.rolling(60).corr(u)
def f_rev_5d_voladj():
    return -(close/close.shift(5)-1) / ret.rolling(20).std()

CANDIDATES = [
    ('parkinson_vol_20d', 'Parkinson vol 20d (inverse vol quality)', f_parkinson_vol_20, -1),
    ('range_ratio_20d', 'Mean daily range / close 20d (inverse vol)', f_range_ratio_20, -1),
    ('max_dd_60d', 'Rolling 60d max drawdown (quality)', f_max_dd_60, 1),
    ('downside_vol_20d', 'Downside deviation 20d (inverse)', f_downside_vol_20, -1),
    ('efficiency_ratio_20d', 'Kaufman efficiency ratio 20d', f_efficiency_ratio_20, 1),
    ('stochastic_pos_20d', '20d stochastic position', f_stochastic_pos_20, 1),
    ('win_rate_20d', '20d win rate (up-day fraction)', f_win_rate_20, 1),
    ('avg_gain_loss_20d', '20d avg gain / avg loss', f_avg_gain_loss_20, 1),
    ('volume_trend_10_60', 'Volume trend 10/60', f_volume_trend_10_60, 1),
    ('amihud_illiq_20d', 'Amihud illiquidity 20d (inverse)', f_amihud_20, -1),
    ('dxy_beta_60d', 'Beta to DXY 60d', f_dxy_beta, -1),
    ('vix_beta_60d', 'Beta to VIX 60d', f_vix_beta, -1),
    ('wti_beta_60d', 'Beta to WTI 60d', f_wti_beta, 1),
    ('xau_beta_60d', 'Beta to XAU 60d', f_xau_beta, -1),
    ('corr_us10y_60d', 'Rolling corr with US10Y 60d', f_corr_us10y_60, -1),
    ('rev_5d_voladj', 'Vol-adj 5d reversal', f_rev_5d_voladj, 1),
]

def fast_ic_series(fac, fwd):
    """fac, fwd: (T,N) arrays -> per-date spearman IC (rankdata then pearson)."""
    out = np.full(T, np.nan)
    for t in range(T):
        f = fac[t]; g = fwd[t]
        m = ~(np.isnan(f) | np.isnan(g))
        if m.sum() < MIN_VALID:
            continue
        rf = rankdata(f[m]); rg = rankdata(g[m])
        rf = rf - rf.mean(); rg = rg - rg.mean()
        denom = np.sqrt((rf*rf).sum() * (rg*rg).sum())
        out[t] = (rf*rg).sum() / denom if denom > 0 else np.nan
    return out

def evaluate(name, fac_np, exp_sign):
    ics = {}
    for h in HORIZONS:
        ics[h] = fast_ic_series(fac_np, FR[h])
    ic10 = ics[H_ADMIT]
    valid = ~np.isnan(ic10)
    n = int(valid.sum())
    ic_mean = float(ic10[valid].mean()) if n else np.nan
    ic_std = float(ic10[valid].std()) if n else np.nan
    icir = ic_mean/ic_std if ic_std and ic_std > 0 else 0.0
    hit = float((ic10[valid]*np.sign(exp_sign) > 0).mean()) if n else np.nan
    cov_ad = float(np.mean(~np.isnan(fac_np)))
    cov_d8 = float(np.mean(np.sum(~np.isnan(fac_np), axis=1) >= MIN_VALID))
    trs = []
    for i in range(T - 10):
        a = fac_np[i]; b = fac_np[i+10]
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
    res = dict(name=name, exp_sign=exp_sign, ic=ic_mean, icir=icir, ic_hit=hit, n_dates=n,
               ic_std=ic_std, cov_ad=cov_ad, cov_d8=cov_d8, turn=turn,
               max_rho=float(best_rho), max_rho_f=best_f,
               decay={str(h): round(float(np.nanmean(ics[h])), 4) for h in HORIZONS})
    for win in [250, 500, 750]:
        if n >= win:
            v = ic10[valid][-win:]
            res[f'ic_r{win}'] = float(v.mean())
            res[f'icir_r{win}'] = float(v.mean()/v.std()) if v.std() > 0 else 0.0
    return res

print('\n=== ACTIVE LIBRARY RE-VALIDATION (through 2030-05-20) ===', flush=True)
for name, fn in [('vol_adj_mom_accel_20x60', f_vol_adj_mom_accel),
                 ('dn_mkt_beta_60d', f_dn_mkt_beta),
                 ('rate_beta_cn10y_60d', f_rate_beta_cn10y)]:
    r = evaluate(name, fn().values, 1)
    print(f"[active] {name}: IC={r['ic']:.4f} ICIR={r['icir']:.4f} hit={r['ic_hit']:.3f} n={r['n_dates']} cov_d8={r['cov_d8']:.3f} r250={r.get('ic_r250',np.nan):.4f}/{r.get('icir_r250',np.nan):.3f} decay10={r['decay']['10']:.4f}", flush=True)

print('\n=== CANDIDATE FACTORS (h=10 admission) ===', flush=True)
results = []
for name, desc, fn, esign in CANDIDATES:
    fac = fn().values
    r = evaluate(name, fac, esign)
    results.append((name, desc, esign, r))
    flag = 'PASS' if (abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084 and r['n_dates'] >= 200) else 'fail'
    print(f"[{flag}] {name} | IC={r['ic']:.4f} ICIR={r['icir']:.4f} hit={r['ic_hit']:.3f} n={r['n_dates']} cov_ad={r['cov_ad']:.3f} cov_d8={r['cov_d8']:.3f} turn={r['turn']:.2f} |rho_lib|={abs(r['max_rho']):.3f}({r['max_rho_f']}) r250={r.get('ic_r250',np.nan):.4f}/{r.get('icir_r250',np.nan):.3f} decay={r['decay']}", flush=True)

print('\n=== SUMMARY TABLE ===', flush=True)
print(f"{'factor':26s} {'IC':>8s} {'ICIR':>8s} {'hit':>6s} {'n':>5s} {'covD8':>6s} {'turn':>6s} {'rhoLib':>7s} {'status':>6s}")
for name, desc, esign, r in results:
    status = 'PASS' if (abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084 and r['n_dates'] >= 200) else ''
    print(f"{name:26s} {r['ic']:8.4f} {r['icir']:8.4f} {r['ic_hit']:6.3f} {r['n_dates']:5d} {r['cov_d8']:6.3f} {r['turn']:6.2f} {abs(r['max_rho']):7.3f} {status:6s}")

# save results for later persistence step
import pickle
with open('scripts/_miner1_20300520_results.pkl','wb') as fh:
    pickle.dump({'results': [(n,d,e,r) for n,d,e,r in results]}, fh)
print('saved results pkl', flush=True)
