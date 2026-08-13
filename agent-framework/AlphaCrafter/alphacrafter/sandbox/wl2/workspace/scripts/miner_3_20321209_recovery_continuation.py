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
# Recovery continuation: lagged 10d residual trend, scaled by 20d risk,
# activated only when lagged VIX impulse is negative and downside breadth is easing.
trend=res.rolling(10,min_periods=8).sum().shift(1)
vol=R.rolling(20,min_periods=10).std().shift(1)
vi=V.pct_change(5).shift(1)
breadth=(R<0).mean(axis=1).rolling(5,min_periods=3).mean().shift(1)
easing=(breadth.diff(3)<0).astype(float)
calm=(vi<0).astype(float)
gate=(0.5+0.5*calm)*(0.5+0.5*easing)
f=(trend/vol).mul(gate,axis=0).replace([np.inf,-np.inf],np.nan)
print('assets',len(D),'calendar_dates',len(C),'active_dates',int((gate>0.5).sum()),'coverage',round(f.notna().sum(axis=1).replace(0,np.nan).mean()/len(D),4))
for h in [1,3,5,10]:
 fr=R.rolling(h).sum().shift(-h); a=[]; dates=[]
 for d in f.index:
  q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   a.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank())); dates.append(d)
 a=pd.Series(a,index=dates).dropna(); print('H',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
 if h==1:
  k=len(a)//2; print('early',round(a.iloc[:k].mean(),6),'late',round(a.iloc[k:].mean(),6))
f.to_csv('scripts/miner_3_20321209_recovery_continuation_signal.csv',index_label='date')
