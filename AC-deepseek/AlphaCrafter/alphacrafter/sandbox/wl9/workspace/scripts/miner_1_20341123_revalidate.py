"""
miner_1 2034-11-23: Re-validate current factor library and explore new candidates.
Current date 2034-11-23, VISIBLE data through previous trading day 2034-11-22.
Metrics: rank IC / ICIR vs forward 10d return; gates |IC|>=0.0070, |ICIR|>=0.084.
"""
import numpy as np, pandas as pd
from pathlib import Path
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
VISIBLE_END = '2034-11-22'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
RESEARCH_START = '2020-03-01'

closes = {}
for a in ASSETS:
    df = pd.read_csv(STOCK_DIR / f'{a}.csv', parse_dates=['date'])
    df = df[df['date'] <= VISIBLE_END].sort_values('date')
    closes[a] = df.set_index('date')['close'].astype(float)
rets = pd.DataFrame({a: closes[a].pct_change() for a in ASSETS}).dropna()
rets = rets[rets.index >= RESEARCH_START]
print(f"Panel: {rets.shape[0]} dates x {rets.shape[1]} assets from {rets.index[0]:%Y-%m-%d} to {rets.index[-1]:%Y-%m-%d}")

hl = {}
for a in ASSETS:
    df = pd.read_csv(STOCK_DIR / f'{a}.csv', parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= VISIBLE_END].set_index('date')
    hl[a] = df[['open','close','high','low','volume']].astype(float)

vix = pd.read_csv(INDEX_DIR / 'VIX.csv', parse_dates=['date'])
vix = vix[vix['date'] <= VISIBLE_END].set_index('date')['close'].astype(float)

macros = {}
for m in ['DXY','USDCNY','USDJPY','EURUSD']:
    df = pd.read_csv(INDEX_DIR / f'{m}.csv', parse_dates=['date'])
    df = df[df['date'] <= VISIBLE_END].set_index('date')['close'].astype(float)
    macros[m] = df

def frame(build):
    out = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        out[a] = build(a).reindex(rets.index)
    return out

def icstats(fv, fwd, min_dates=8):
    ics=[]
    for d in sorted(set(fv.index)&set(fwd.index)):
        f=fv.loc[d]; r=fwd.loc[d]; ok=f.notna()&r.notna()
        if ok.sum()>=min_dates:
            fv_=f[ok].rank().values; rv_=r[ok].rank().values
            if np.std(fv_)>0 and np.std(rv_)>0:
                ics.append(np.corrcoef(fv_,rv_)[0,1])
    ics=np.array(ics)
    if len(ics)<20: return {'IC':0.0,'ICIR':0.0,'n':len(ics),'hit':0.0}
    return {'IC':float(ics.mean()),'ICIR':float(ics.mean()/ics.std()*len(ics)**.5) if ics.std()>0 else 0.0,
            'n':len(ics),'hit':float((ics>0).mean())}

def turnover(fv):
    r = fv.rank(axis=1)
    s = np.sign(r.sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

fwd = rets.shift(-10).rolling(10).mean()
vr = vix.pct_change(); dr = macros['DXY'].pct_change(); cr = macros['USDCNY'].pct_change()
ujr = macros['USDJPY'].pct_change(); eur = macros['EURUSD'].pct_change()

F = {}
# current library
F['mom_10d_skip5']=frame(lambda a: closes[a].pct_change(15))
F['mom_120d_skip5']=frame(lambda a: closes[a].pct_change(125))
def vb(a,w):
    j=pd.concat([rets[a].rename('a'),vr.rename('v')],axis=1).dropna()
    return j['a'].rolling(w).cov(j['v'])/j['v'].rolling(w).var()
F['beta_VIX_60']=frame(lambda a: vb(a,60))
vroc=vix.pct_change(20)
F['vix_roc_20d']=frame(lambda a: (vroc if a in ['XAU','US10Y','CN10Y'] else -vroc))
F['mom_10_vixreg']=frame(lambda a: closes[a].pct_change(5)*np.sign(vix.pct_change(10).shift(5)))
F['ac1_120d']=frame(lambda a: rets[a].rolling(120).apply(lambda x: pd.Series(x).autocorr(1) if len(x)>=30 else np.nan, raw=False))
F['kaufman_eff_20d']=frame(lambda a: closes[a].diff(20).abs()/closes[a].diff().abs().rolling(20).sum())
F['bb_width_20d']=frame(lambda a: (closes[a]-closes[a].rolling(20).mean())/closes[a].rolling(20).std())
def sk(a):
    r=rets[a]; m=r.rolling(20).mean(); sd=r.rolling(20).std()
    return ((r-m)**3).rolling(20).mean()/sd**3
F['skew_20d']=frame(sk)
F['vol_z_20d']=frame(lambda a: (hl[a]['volume']-hl[a]['volume'].rolling(20).mean())/hl[a]['volume'].rolling(20).std())
F['rng_pos_20d']=frame(lambda a: (hl[a]['close']-hl[a]['low'].rolling(20).min())/(hl[a]['high'].rolling(20).max()-hl[a]['low'].rolling(20).min()))
F['streak_len_14']=frame(lambda a: ((closes[a].diff()>0).astype(int).groupby(((closes[a].diff()>0).astype(int)!=(closes[a].diff()>0).astype(int).shift()).cumsum()).cumcount()+1))
def dsh(a):
    c=closes[a]; ah=(c==c.rolling(60).max()); return ah.groupby((~ah).cumsum()).cumcount()
F['days_since_high_60']=frame(dsh)
F['cny_beta_60']=frame(lambda a: (rets[a].rolling(60).cov(cr)/cr.rolling(60).var()))
def dxycc(a):
    j=pd.concat([rets[a].rename('a'),dr.rename('d')],axis=1).dropna()
    return j['a'].rolling(20).corr(j['d'])-