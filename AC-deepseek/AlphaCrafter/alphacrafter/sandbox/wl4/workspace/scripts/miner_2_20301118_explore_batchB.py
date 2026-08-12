"""miner_2 exploration batch B - 2030-11-18.
Test more decorrelated families + orthogonalization of promising candidates.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

ASSETS = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
VIS = '2030-11-15'
FROZEN = {'000300.SH','HSI','ETH'}

def load(sym, col='close'):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv', parse_dates=['date'])
    return df[df['date'] <= VIS].set_index('date')[col]

px = pd.DataFrame({s: load(s) for s in ASSETS}).dropna(how='all')
ret = px.pct_change()
mkt = ret.mean(axis=1)
open_ = pd.DataFrame({s: load(s,'open') for s in ASSETS})
high = pd.DataFrame({s: load(s,'high') for s in ASSETS})
low = pd.DataFrame({s: load(s,'low') for s in ASSETS})
vol = pd.DataFrame({s: load(s,'volume') for s in ASSETS})

# library
mom20 = px/px.shift(20)-1; mom60 = px/px.shift(60)-1; v20 = ret.rolling(20).std()
f_mom = (mom20-mom60)/v20
down = ret.where(mkt<0)
f_dn = down.rolling(60,min_periods=40).cov(mkt).div(mkt.where(mkt<0).rolling(60,min_periods=40).var(), axis=0)
dcn = px['CN10Y'].pct_change()
f_rate = ret.rolling(60,min_periods=40).cov(dcn).div(dcn.rolling(60,min_periods=40).var(), axis=0)
lib = {'vol_adj_mom_accel_20x60': f_mom, 'dn_mkt_beta_60d': f_dn, 'rate_beta_cn10y_60d': f_rate}

# US10Y & spread betas
us10y = px['US10Y']; dus = us10y.pct_change()
spread = px['US10Y'] - px['CN10Y']; dsp = spread.pct_change()

cands = {}
# close location value: (close-low)/(high-low), mean 20d (intraday sentiment)
rng = (high-low).replace(0, np.nan)
cands['clv_20'] = ((px-low)/rng).rolling(20).mean()
# OBV-style: cumsum(sign(ret)*volume) / rolling sum volume -> volume flow
obv = (np.sign(ret)*vol).rolling(20).sum() / vol.rolling(20).sum()
cands['obv_flow_20'] = obv
# leverage effect: rolling corr(ret_t, vol_{t-1}) (vol feedback)
cands['lev_eff_60'] = ret.rolling(60).corr(ret.rolling(20).std().shift(1))
# up-down beta asymmetry
up = ret.where(mkt>0)
f_up = up.rolling(60,min_periods=40).cov(mkt).div(mkt.where(mkt>0).rolling(60,min_periods=40).var(), axis=0)
cands['updown_beta_asym_60'] = f_up - f_dn
# US10Y beta 60
cands['us10y_beta_60'] = ret.rolling(60,min_periods=40).cov(dus).div(dus.rolling(60,min_periods=40).var(), axis=0)
# spread beta 60
cands['spread_beta_60'] = ret.rolling(60,min_periods=40).cov(dsp).div(dsp.rolling(60,min_periods=40).var(), axis=0)
# drawdown duration: count of days since 60d high
cands['ddur_60'] = px.rolling(60).apply(lambda x: np.argmax(x[::-1]) if len(x)==60 else np.nan, raw=True)
# hurst-like: log(R/S) coarse via variance ratio
def hurst_est(x):
    if len(x)<40 or not np.isfinite(x).all(): return np.nan
    x = x - x.mean()
    # variance ratio: var of 5d sums / (5*var of 1d)
    s1 = x.var()
    if s1 <= 0: return np.nan
    s5 = x.reshape(-1,5).sum(axis=1).var() if len(x)%5==0 else pd.Series(x).rolling(5).sum().dropna().var()
    return np.log(max(s5/s1,1e-8)/5)/(2*np.log(5))
cands['hurst_60'] = ret.rolling(60).apply(hurst_est, raw=True)
# downside beta to US10Y (conditional)
dus_neg = dus.where(dus<0)
cands['us10y_dn_beta_60'] = ret.rolling(60,min_periods=40).cov(dus_neg).div(dus_neg.rolling(60,min_periods=40).var(), axis=0)
# vol of price zscore change (reversal intensity)
cands['zchg_20'] = (px.pct_change(5)).rolling(20).std() / ret.rolling(20).std()
# 1d reversal after big moves: -sign(ret)*|ret|/vol, smoothed
cands['bigmove_rev_20'] = (-np.sign(ret)*ret.abs()/v20).rolling(20).mean()

H = 10
fwd = px.shift(-H)/px - 1

def ic_series(sig, fwd, frozen=FROZEN, warm=80):
    out=[]
    for t in sig.index[warm:-H]:
        s=sig.loc[t].dropna(); fr=fwd.loc[t].dropna()
        common=[c for c in s.index.intersection(fr.index) if c not in frozen]
        if len(common)<8: continue
        ic,_=spearmanr(s[common], fr[common])
        if np.isfinite(ic): out.append((t,ic))
    return pd.Series([x[1] for x in out], index=[x[0] for x in out])

def lib_corr(sig, dates, frozen=FROZEN):
    corrs=[]
    for t in dates:
        s=sig.loc[t].dropna()
        for ln, ls in lib.items():
            l=ls.loc[t].dropna()
            common=[c for c in s.index.intersection(l.index) if c not in frozen]
            if len(common)>=8:
                rho,_=spearmanr(s[common], l[common])
                if np.isfinite(rho): corrs.append((ln, abs(rho)))
    return corrs

print(f"{'factor':26s} {'fullIC':>8s} {'fullICIR':>8s} {'n':>5s} {'r180IC':>7s} {'r180ICIR':>8s} {'maxRho':>7s} {'rhoWith':22s} {'gate':>5s}")
for name, sig in cands.items():
    ic = ic_series(sig, fwd)
    if len(ic)==0:
        print(f'{name:26s} no data'); continue
    recent = ic[ic.index >= ic.index[-1] - pd.Timedelta(days=400)].tail(180)
    dates = sig.index[100:-10][::7]
    cr = lib_corr(sig, dates)
    maxc = max((v for _,v in cr), default=np.nan); maxcf = max(cr, key=lambda x:x[1])[0] if cr else 'NA'
    full_ok = abs(ic.mean())>=0.007 and abs(ic.mean()/ic.std())>=0.084
    recent_ok = abs(recent.mean())>=0.007 and abs(recent.mean()/recent.std())>=0.084
    rho_ok = (not np.isfinite(maxc)) or maxc < 0.5
    gate = 'PASS' if (full_ok and rho_ok) else ('recent-ok' if recent_ok else 'fail')
    print(f'{name:26s} {ic.mean():+8.4f} {ic.mean()/ic.std():+8.3f} {len(ic):5d} {recent.mean():+7.4f} {recent.mean()/recent.std():+8.3f} {maxc:7.3f} {maxcf:22s} {gate}')

# ---- orthogonalization test on best raw candidates ----
print('\n--- orthogonalized residuals (cross-sectional OLS on library factors) ---')
raw_best = {
    'kurt_60': ret.rolling(60).kurt(),
    'downside_ratio_60': ret.rolling(60).mean()/np.sqrt((ret.clip(upper=0)**2).rolling(60).mean()),
    'updown_vol_ratio_60': np.sqrt((ret.clip(lower=0)**2).rolling(60).mean())/np.sqrt((ret.clip(upper=0)**2).rolling(60).mean()),
    'dxy_beta_60': None,
}
def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    dcol = 'date' if 'date' in df.columns else df.columns[0]
    df = df.rename(columns={dcol:'date'}); df['date']=pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()
DXY = load_macro('DXY'); dxy = DXY.iloc[:,0].reindex(px.index).ffill().pct_change()
raw_best['dxy_beta_60'] = ret.rolling(60,min_periods=40).cov(dxy).div(dxy.rolling(60,min_periods=40).var(), axis=0)

libmat = {'L1': f_mom, 'L2': f_dn, 'L3': f_rate}
for name, sig in raw_best.items():
    if sig is None: continue
    resid = sig.copy()
    for t in sig.index:
        row = sig.loc[t]
        X = pd.DataFrame({k: v.loc[t] for k,v in libmat.items()})
        y = row
        d = pd.concat([y.rename('y'), X], axis=1).dropna()
        if len(d) < 8: 
            resid.loc[t] = np.nan; continue
        # cross-sectional OLS
        Xd = d[['L1','L2','L3']].values; yd = d['y'].values
        try:
            coef, *_ = np.linalg.lstsq(np.column_stack([Xd, np.ones(len(Xd))]), yd, rcond=None)
            resid.loc[t] = yd - Xd@coef[:3] - coef[3]
        except Exception:
            resid.loc[t] = np.nan
    ic = ic_series(resid, fwd)
    if len(ic)==0:
        print(name, 'orth residual: no data'); continue
    recent = ic[ic.index >= ic.index[-1]-pd.Timedelta(days=400)].tail(180)
    cr = lib_corr(resid, resid.index[100:-10][::7])
    maxc = max((v for _,v in cr), default=np.nan)
    print(f'{name:26s} ORTH IC {ic.mean():+.4f} ICIR {ic.mean()/ic.std():+.3f} n={len(ic):4d} | r180 IC {recent.mean():+.4f} ICIR {recent.mean()/recent.std():+.3f} | maxRho {maxc:.3f}')
