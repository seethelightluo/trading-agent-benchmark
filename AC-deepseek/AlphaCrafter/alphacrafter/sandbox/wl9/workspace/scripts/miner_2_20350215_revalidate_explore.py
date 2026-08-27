"""
miner_2 2035-02-15: Re-validate all effective factors + explore new candidates.
Visible-through: 2035-02-14. Full history 2020-01-02..2035-02-14.
Gates: abs(paper IC)>=0.0070 AND abs(paper ICIR)>=0.0840 at 10d, min 8 assets/date.
"""
import numpy as np, pandas as pd, json, zlib, base64, hashlib
from pathlib import Path
VISIBLE_END='2035-02-14'
SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets,end):
    closes={}
    for a in assets:
        f=SD/f'{a}.csv'
        if not f.exists(): f=ID/f'{a}.csv'
        df=pd.read_csv(f,parse_dates=['date'])
        df=df[df['date']<=pd.Timestamp(end)].sort_values('date').set_index('date')
        closes[a]=df['close'].astype(float)
    return pd.DataFrame(closes).dropna(axis=1,how='all')

close=load(ASSETS,VISIBLE_END)
print(f"Panel {close.shape[0]} dates x {close.shape[1]} assets, {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}",flush=True)
rets=close.pct_change()

def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5,fwd10,fwd20=fwd(5),fwd(10),fwd(20)

def compute_ic(fv,fw,min_dates=30,start=None):
    f=fv.reindex(fw.index); ii=fw.index
    if start: ii=ii[ii>=pd.Timestamp(start)]
    ics=[];ok=0
    for d in ii:
        x=f.loc[d]; y=fw.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0:
                ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.,cov=0.,ok=ok)
    mu=ics.mean();sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),
                hit=float((ics>0).mean()),cov=float(f.notna().mean().mean()),ok=ok)

def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name,fv,start=None):
    a=compute_ic(fv,fwd10,start=start)
    b=compute_ic(fv,fwd5,start=start); c=compute_ic(fv,fwd20,start=start)
    ok=(abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084)
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok_d={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} | [5]{b['IC']:.3f} [20]{c['IC']:.4f}",flush=True)
    return a,ok

# Load macro
vix=load(['VIX.none'],VISIBLE_END) if False else pd.read_csv(ID/'VIX.csv',parse_dates=['date'])
mdf={}
for m in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    df=pd.read_csv(ID/f'{m}.csv',parse_dates=['date'])
    df=df[df['date']<=pd.Timestamp(VISIBLE_END)].sort_values('date').set_index('date')
    mdf[m]=df['close'].astype(float)
vix_r=mdf['VIX'].pct_change()
dxy_r=mdf['DXY'].pct_change()
cny_r=mdf['USDCNY'].pct_change()
usdjpy_r=mdf['USDJPY'].pct_change()

def beta_macro(r_asset, m_r, window):
    j=pd.concat([r_asset.rename('a'),m_r.rename('m')],axis=1).dropna()
    return j['a'].rolling(window).cov(j['m'])/j['m'].rolling(window).var()

print('\n=== RE-VALIDATE EXISTING FACTORS (2033-01-01 recent sample) ===',flush=True)
# 1 vol_z_20d
def f_vol_z():
    df={}
    for a in close.columns:
        v=pd.read_csv(SD/f'{a}.csv',parse_dates=['date']) if (SD/f'{a}.csv').exists() else pd.read_csv(ID/f'{a}.csv',parse_dates=['date'])
        v=v[v['date']<=pd.Timestamp(VISIBLE_END)].sort_values('date').set_index('date')
        if 'volume' in v and v['volume'].notna().any():
            s=v['volume'].astype(float)
            df[a]=(s-s.rolling(20).mean())/s.rolling(20).std()
    return pd.DataFrame(df).reindex(close.index)
report('vol_z_20d', z_volz:=f_vol_z(), start='2033-01-01')

# 2 bb_width_20d
bbw=(close/close.rolling(20).mean()-1).rolling(20).std()*4/ (close.rolling(20).std()/close.rolling(20).mean()+1e-12)
bbw=(close.rolling(20).std()/close.rolling(20).mean()*4)
report('bb_width_20d', z_bbw:=bbw, start='2033-01-01')

# 3 mom_10d_skip5
report('mom_10d_skip5', (close.rolling(15).apply(lambda x:x[-1]/x[0]-1,raw=True),True)[0] if False else close.shift(5)/close.shift(15)-1, start='2033-01-01')
report('mom_120d_skip5', close.shift(5)/close.shift(125)-1, start='2033-01-01')

# 4 kaufman eff 20d
kauf=(close-close.shift(20)).abs()/(close.diff().abs().rolling(20).sum()+1e-12)
report('kaufman_eff_20d', kauf, start='2033-01-01')

# 5 ac1_120d
def f_ac1(w=120):
    j=rets.copy()
    out=pd.DataFrame(index=rets.index,columns=ASSETS,dtype=float)
    for a in ASSETS:
        r=rets[a]
        out[a]=r.rolling(w).apply(lambda x: np.corrcoef(x[:-1],x[1:])[0,1] if len(x)>3 else np.nan, raw=True)
    return out
ac1=f_ac1()
report('ac1_120d', ac1, start='2033-01-01')

# 6 beta_VIX_60
def f_beta(m_r,win):
    out=pd.DataFrame(index=rets.index,columns=ASSETS,dtype=float)
    for a in ASSETS:
        j=pd.concat([