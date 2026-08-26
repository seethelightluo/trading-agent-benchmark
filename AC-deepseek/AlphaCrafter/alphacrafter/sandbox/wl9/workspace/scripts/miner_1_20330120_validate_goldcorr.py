"""miner_1 cycle 2033-01-20: validate corr_to_gold_60 candidate + library correlation.
Visible history up to 2033-01-19. No lookahead.
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2033-01-19'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load_series(assets):
    out = {}
    for a in assets:
        f = STOCK_DIR/f'{a}.csv'
        if not f.exists(): f = INDEX_DIR/f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date']<=VISIBLE_END].sort_values('date').set_index('date')
        s = df['close'].astype(float)
        s = s[~s.index.duplicated(keep='last')]
        out[a] = s
    return out

ser = load_series(ASSETS)
close = pd.DataFrame(ser)
# align on common dates
close = close.dropna()
rets = close.pct_change().dropna()
fwd10 = rets.shift(-10).rolling(10).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}", flush=True)

def compute_ic(fv, fwd, min_dates=60):
    fv = fv.reindex(fwd.index)
    ics=[]; n_dates=0
    for d in fwd.index:
        f=fv.loc[d]; r=fwd.loc[d]
        m=f.notna()&r.notna()
        if m.sum()>=8:
            n_dates+=1
            fv_=f[m].rank().values; rv_=r[m].rank().values
            if fv_.std()>0 and rv_.std()>0: ics.append(np.corrcoef(fv_,rv_)[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return {'IC':0.0,'ICIR':0.0,'n':len(ics),'hit':0.0,'cov':0.0}
    mu=ics.mean(); sd=ics.std()
    icir = mu/sd*np.sqrt(len(ics)) if sd>0 else 0.0
    hit=float((ics>0).mean()); cov=float(fv.notna().mean().mean())
    return {'IC':float(mu),'ICIR':float(icir),'n':len(ics),'hit':hit,'cov':cov}

def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

# ---- candidate B: corr-to-gold linkage, 10d returns corr with XAU over 60d ----
xau10 = close.pct_change(10)['XAU']
cand = pd.DataFrame({a: close.pct_change(10)[a].rolling(60).corr(xau10) for a in ASSETS}).reindex(fwd10.index)
ic = compute_ic(cand, fwd10)
print(f"CORR_TO_GOLD_60: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
      f"hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(cand):.3f}", flush=True)

# ---- library correlation: pairwise rho of cross-sectional RANK signals ----
try:
    vix = load_series(['VIX'])['VIX']
except Exception:
    vix = pd.read_csv(INDEX_DIR/'VIX.csv', parse_dates=['date'])
    vix = vix[vix['date']<=VISIBLE_END].set_index('date')['close']; vix=vix[~vix.index.duplicated(keep='last')]
try:
    dxy = load_series(['DXY'])['DXY']
except Exception:
    dxy = pd.read_csv(INDEX_DIR/'DXY.csv', parse_dates=['date'])
    dxy = dxy[dxy['date']<=VISIBLE_END].set_index('date')['close']; dxy=dxy[~dxy.index.duplicated(keep='last')]

c=close
def library_factor(fid):
    if fid=='kaufman_eff_20d':
        return c.diff().abs().rolling(20).mean()/c.pct_change().abs().rolling(20).sum()
    if fid=='mom_120d_skip5':
        return c.shift(5)/c.shift(125)-1
    if fid=='mom_10d_skip5':
        return c.shift(5)/c.shift(15)-1
    if fid=='bb_width_20d':
        return (c.rolling(20).max()-c.rolling(20).min())/c.rolling(20).mean()
    if fid=='vol_z_20d':
        v=c.pct_change().rolling(20).std(); return (v-v.rolling(120).mean())/v.rolling(120).std()
    if fid=='skew_20d':
        return c.pct_change().rolling(20).skew()
    if fid=='kurt_20d':
        return c.pct_change().rolling(20).kurt()
    if fid=='ac1_120d':
        r=c.pct_change(); return r.rolling(120).apply(lambda x: np.corrcoef(x[:-1],x[1:])[0,1],raw=True)
    if fid=='beta_VIX_60':
        vr=vix.pct_change().dropna(); rr=close.pct_change().dropna()
        return rr.rolling(60).cov(vr)/vr.rolling(60).var()
    if fid=='vix_beta_cond_60x20':
        vr=vix.pct_change().dropna(); rr=close.pct_change().dropna()
        beta=rr.rolling(60).cov(vr)/vr.rolling(60).var()
        cond=(vix.pct_change(20)>0).astype(float)
        return beta*cond.reindex(beta.index)
    if fid=='dxy_corr_change_20_60':
        dr=dxy.pct_change(10); r10=c.pct_change(10)
        return r10.rolling(20).corr(dr)-r10.rolling(60).corr(dr)
    if fid=='cny_beta_60':
        # use CN10Y as risk proxy beta 60d
        cn10=c['CN10Y']; cr=close.pct_change().dropna(); ref=cn10.pct_change().dropna()
        return cr.rolling(60).cov(ref)/ref.rolling(60).var()
    if fid=='days_since_high_60':
        h=c.rolling(60).max()
        return (c==h).astype(int)  # proxy (1 day since high)
    if fid=='rng_pos_20d':
        rng=(c.rolling(20).max()-c.rolling(20).min())/c
        return rng.rolling(120).rank(pct=True)
    if fid=='streak_len_14':
        r=(c.pct_change()>0).astype(int); out=pd.DataFrame(0.0,index=c.index,columns=c.columns)
        for col in c.columns:
            s=r[col]; grp=(s!=s.shift()).cumsum(); cnt=s.groupby(grp).cumsum()
            out[col]=cnt
        return out
    return None

lib = {}
for fid in ['beta_VIX_60','kaufman_eff_20d','mom_120d_skip5','bb_width_20d','cny_beta_60',
            'vol_z_20d','ac1_120d','mom_10d_skip5','dxy_corr_change_20_60','skew_20d