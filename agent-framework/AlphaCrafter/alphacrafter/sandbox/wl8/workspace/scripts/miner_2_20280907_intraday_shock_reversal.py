import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').sort_index()
px=pd.concat({s:d.close.astype(float) for s,d in D.items()},axis=1).sort_index()
hi=pd.concat({s:d.high.astype(float) for s,d in D.items()},axis=1).reindex(px.index)
lo=pd.concat({s:d.low.astype(float) for s,d in D.items()},axis=1).reindex(px.index)
r=px.pct_change()
# Candidate: 3-session reversal, normalized by lagged 20d vol and activated by
# an unusually wide prior-day range (bounded shock multiplier).
rv=r.rolling(20).std().shift(1)
rng=(hi/lo-1).replace([np.inf,-np.inf],np.nan)
z=(rng.shift(1)-rng.shift(1).rolling(60).median())/(rng.shift(1).rolling(60).std()+1e-12)
f=-(px.shift(1)/px.shift(4)-1)/(rv+1e-12)*(1+0.35*z.clip(-1.5,1.5))
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for i,dt in enumerate(px.index):
 for h in [1,3,5,10]:
  if i+h>=len(px): continue
  q=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8: rows.append((dt,h,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
o=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('assets',px.shape[1],'dates',px.index.min(),px.index.max(),'rows',len(o))
for h in [1,3,5,10]:
 q=o[o.h==h].ic.dropna(); z2=o[o.h==h]
 print(h,'dates',len(q),'avgN %.2f IC %.6f ICIR %.6f hit %.3f'%(z2.n.mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 for lab,mask in [('2020-22',z2.date.dt.year<=2022),('2023-25',z2.date.dt.year.between(2023,2025)),('2026',z2.date.dt.year==2026),('2027',z2.date.dt.year==2027),('2028',z2.date.dt.year==2028),('recent280',z2.date>=z2.date.max()-pd.Timedelta(days=280))]:
  a=z2[mask].ic.dropna()
  if len(a): print(' ',lab,len(a),'IC %.6f ICIR %.6f'%(a.mean(),a.mean()/a.std(ddof=1)))
f.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_2_20280907_intraday_shock_reversal_signal.csv',index=False)
