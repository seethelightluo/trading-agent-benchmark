import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    try: d=get_index_daily_data(s, days=5000)
    except Exception:
        try: d=get_stock_daily_data(s, days=5000)
        except Exception: d=None
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date')['close'].astype(float).sort_index()
        frames[s]=x
p=pd.DataFrame(frames).sort_index().ffill()
# Candidate: continuous dispersion-conditioned trend/reversal. All stats lagged one day.
r5=p.pct_change(5); r20=p.pct_change(20)
disp=r5.std(axis=1).shift(1)
# cross-sectional dispersion percentile over 252 completed days; smooth gate 0..1
q=disp.rolling(252,min_periods=100).rank(pct=True).shift(1)
# trend in low dispersion, reversal in high dispersion, with smooth transition around 65th pct
w=((q-0.65)/0.15).clip(0,1)
f=r20.shift(1).mul(1-w,axis=0) + r5.shift(1).mul(-w,axis=0)
# forward returns from t close to t+10 close, factor at t
fr=p.shift(-10)/p-1
rows=[]
for dt in f.index:
    a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); x.index=pd.to_datetime(x.index)
def stat(y):
    return len(y), y.ic.mean(), y.ic.mean()/y.ic.std(ddof=1) if len(y)>1 and y.ic.std(ddof=1)>0 else np.nan, (y.ic>0).mean(), y.n.mean()
print('data',p.index.min(),p.index.max(),'dates',len(x),'avgN',x.n.mean())
print('full n/IC/ICIR/hit/avgN',stat(x))
for n in [365,180,60]: print('recent',n,stat(x.tail(n)))
for yr,g in x.groupby(x.index.year): print('year',yr,stat(g))
print('decay')
for h in [1,5,10,20]:
    ff=p.shift(-h)/p-1; rr=[]
    for dt in f.index:
      z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
      if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    print(h,len(rr),np.mean(rr))
# turnover rank signal
r=f.rank(axis=1,pct=True); turn=(r-r.shift(1)).abs().mean(axis=1).dropna(); print('turnover',turn.mean(),'coverage',p.notna().mean().mean())
# save artifacts for provenance
sig=f.loc[x.index].copy(); sig.index.name='date'; sig.to_csv('scripts/miner_2_20320205_continuous_dispersion_signal.csv')
x.to_csv('scripts/miner_2_20320205_continuous_dispersion_ic.csv')
