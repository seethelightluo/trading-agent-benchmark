"""miner_2 exploration batch A - 2030-11-18 (data visible through 2030-11-15).
Goal: find factor ideas passing IC/ICIR gates AND decorrelated from the 3-factor library.
Library: vol_adj_mom_accel_20x60, dn_mkt_beta_60d, rate_beta_cn10y_60d.
Gates: |IC| >= 0.007, |ICIR| >= 0.084, max_abs_library_correlation < 0.5.
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

# --- library reference implementations ---
mom20 = px/px.shift(20)-1; mom60 = px/px.shift(60)-1; v20 = ret.rolling(20).std()
f_mom = (mom20-mom60)/v20
down = ret.where(mkt<0)
f_dn = down.rolling(60,min_periods=40).cov(mkt).div(mkt.where(mkt<0).rolling(60,min_periods=40).var(), axis=0)
dcn = px['CN10Y'].pct_change()
f_rate = ret.rolling(60,min_periods=40).cov(dcn).div(dcn.rolling(60,min_periods=40).var(), axis=0)
lib = {'vol_adj_mom_accel_20x60': f_mom, 'dn_mkt_beta_60d': f_dn, 'rate_beta_cn10y_60d': f_rate}

# --- observation-only macro (normalized daily changes) ---
def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=[0] if False else None)
    df.columns = [c.strip().lower() for c in df.columns]
    dcol = 'date' if 'date' in df.columns else df.columns[0]
    df = df.rename(columns={dcol:'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()

DXY = load_macro('DXY'); VIX = load_macro('VIX'); EURUSD = load_macro('EURUSD')
USDCNY = load_macro('USDCNY'); USDJPY = load_macro('USDJPY')
dxy = DXY.iloc[:,0].reindex(px.index).ffill().pct_change()
vix = VIX.iloc[:,0].reindex(px.index).ffill().pct_change()
usdjpy = USDJPY.iloc[:,0].reindex(px.index).ffill().pct_change()

# --- candidate factors ---
cands = {}
# downside deviation (proper): sqrt(mean(min(ret,0)^2))
downside_dev = np.sqrt((ret.clip(upper=0)**2).rolling(60).mean())
upside_dev = np.sqrt((ret.clip(lower=0)**2).rolling(60).mean())
cands['downside_ratio_60'] = ret.rolling(60).mean() / downside_dev
cands['updown_vol_ratio_60'] = upside_dev / downside_dev
cands['tail_risk_60'] = ret.rolling(60).min()
cands['max_gain_60'] = ret.rolling(60).max()
cands['pos_ratio_20'] = (ret>0).rolling(20).mean()
cands['close_open_20'] = (close_ := px/open_ - 1).rolling(20).mean()
cands['gap_20'] = (open_/px.shift(1) - 1).rolling(20).mean()
cands['vol_ratio_10_60'] = ret.rolling(10).std() / ret.rolling(60).std()
cands['vol_level_60'] = -ret.rolling(60).std()
cands['vol_trend_10_60'] = ret.rolling(10).std() / ret.rolling(60).std()  # alias
cands['volume_trend_10_60'] = vol.rolling(10).mean() / vol.rolling(60).mean()
cands['volume_z_20'] = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
cands['dxy_beta_20'] = ret.rolling(20,min_periods=15).cov(dxy).div(dxy.rolling(20,min_periods=15).var(), axis=0)
cands['dxy_beta_60'] = ret.rolling(60,min_periods=40).cov(dxy).div(dxy.rolling(60,min_periods=40).var(), axis=0)
cands['vix_beta_20'] = ret.rolling(20,min_periods=15).cov(vix).div(vix.rolling(20,min_periods=15).var(), axis=0)
cands['usdjpy_beta_60'] = ret.rolling(60,min_periods=40).cov(usdjpy).div(usdjpy.rolling(60,min_periods=40).var(), axis=0)
cands['high_low_range_60'] = -((high-low)/px).rolling(60).mean()
cands['kurt_60'] = ret.rolling(60).kurt()
# asymmetric momentum: up-day momentum minus down-day momentum
up_ret = ret.where(ret>0, 0.0); dn_ret = ret.where(ret<0, 0.0)
cands['asym_mom_20'] = (up_ret.rolling(20).sum() - dn_ret.rolling(20).sum().abs()) / v20

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
    if cr:
        maxc = max(v for _,v in cr); maxcf = max(cr, key=lambda x:x[1])[0]
    else:
        maxc = np.nan; maxcf = 'NA'
    full_ok = abs(ic.mean())>=0.007 and abs(ic.mean()/ic.std())>=0.084
    recent_ok = abs(recent.mean())>=0.007 and abs(recent.mean()/recent.std())>=0.084
    rho_ok = (not np.isfinite(maxc)) or maxc < 0.5
    gate = 'PASS' if (full_ok and rho_ok) else ('recent-ok' if recent_ok else 'fail')
    print(f'{name:26s} {ic.mean():+8.4f} {ic.mean()/ic.std():+8.3f} {len(ic):5d} {recent.mean():+7.4f} {recent.mean()/recent.std():+8.3f} {maxc:7.3f} {maxcf:22s} {gate}')
