import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            z=fn(s,5000)
            if z is not None and len(z)>100:return z
        except Exception: pass
D={s:load(s) for s in U}; D={s:z for s,z in D.items() if z is not None}
C=pd.DataFrame({s:z.set_index(pd.to_datetime(z.date)).close.astype(float) for s,z in D.items()}).sort_index().groupby(level=0).last()
R=C.ffill().pct_change(); res=R.sub(R.mean(axis=1),axis=0)
v=get_index_daily_data('VIX',5000); V=v.set_index(pd.to_datetime(v.date)).close.astype(float).reindex(C.index).ffill()
# Extreme-stress residual reversal: lagged 5d underperformance, risk scaled,
# activated only in unusually high lagged VIX and broad downside stress.
rv=res.rolling(5,min_periods=4).sum().shift(1)
vol=R.rolling(20,min_periods=10).std().shift(1)
vlag=V.shift(1); q=vlag.rolling(252,min_periods=60).quantile(.75)
breadth=(R<0).mean(axis=1).shift(1)
gate=((vlag>q)&(breadth>=.60)).astype(float)
f=(-rv/vol).mul(gate,axis=0).replace([np.inf,-np.inf],np.nan)
print('assets',len(D),'calendar_dates',len(C),'active_dates',int(gate.sum()),'coverage',round(f.notna().sum(axis=1).replace(0,np.nan).mean()/len(D),4))
for h in [1,3,5,10]:
 fr=R.rolling(h).sum().shift(-h); a=[]; dates=[]
 for d in f.index:
  qv=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(qv)>=8 and qv.iloc[:,0].nunique()>1:
   a.append(qv.iloc[:,0].rank().corr(qv.iloc[:,1].rank())); dates.append(d)
 a=pd.Series(a,index=dates).dropna(); print('H',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
 if h==1 and len(a):
  k=len(a)//2; print('early',round(a.iloc[:k].mean(),6),'late',round(a.iloc[k:].mean(),6))
f.to_csv('scripts/miner_3_20321223_extreme_stress_residual_reversal_signal.csv',index_label='date')
