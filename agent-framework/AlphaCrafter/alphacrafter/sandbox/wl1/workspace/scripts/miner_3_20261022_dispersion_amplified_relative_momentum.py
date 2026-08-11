import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.DataFrame(P).ffill()
# Dispersion-amplified relative momentum: 20d relative strength multiplied by
# lagged cross-sectional dispersion, testing whether differentiated markets reward continuation.
csret=p.pct_change()
csmean=csret.mean(axis=1)
disp=csret.sub(csmean,axis=0).abs().mean(axis=1).rolling(5).mean().shift(1)
scale=(disp/disp.rolling(120).median()).clip(0.25,4.0)
rel=p.div(p.median(axis=1),axis=0)
f=(rel/rel.shift(20)-1).shift(1).mul(scale,axis=0)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(p.index[i])
 a=np.asarray(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 print('annual',h,{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True); print('turnover',round(float((rank-rank.shift(1)).abs().stack().mean()),6),'factor_dates',int(f.notna().sum(axis=1).ge(8).sum()))
