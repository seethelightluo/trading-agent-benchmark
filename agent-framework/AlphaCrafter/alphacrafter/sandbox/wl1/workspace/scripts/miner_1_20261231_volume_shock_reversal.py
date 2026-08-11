import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-30')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
close=pd.DataFrame({s:x.close for s,x in P.items()}); vol=pd.DataFrame({s:x.volume for s,x in P.items()})
idx=close.index.union(vol.index).sort_values(); close=close.reindex(idx).ffill(); vol=vol.reindex(idx).ffill()
r=close.pct_change(); ret5=close/close.shift(5)-1
# Abnormal volume is cross-sectionally normalized log recent/long volume; high-volume losers may rebound.
vr=np.log((vol.rolling(5,min_periods=5).mean()+1e-9)/(vol.rolling(60,min_periods=30).mean()+1e-9))
# demean volume shock across assets to avoid common market volume effect
vr=vr.sub(vr.mean(axis=1),axis=0)
f=(-ret5*vr).shift(1)
f=f.replace([np.inf,-np.inf],np.nan)
print('universe',len(U),'dates',len(close),'cutoff',close.index.max().date())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(close)-h):
  q=pd.concat([f.iloc[i].rename('f'),(close.iloc[i+h]/close.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   I.append(spearmanr(q.f,q.y).statistic);Ns.append(len(q));ds.append(close.index[i])
 a=np.asarray(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10: print('annual10',{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
