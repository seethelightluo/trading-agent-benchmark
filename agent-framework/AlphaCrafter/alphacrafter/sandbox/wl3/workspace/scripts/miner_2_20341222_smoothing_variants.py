import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_index_daily_data(s,6000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None})
px.index=pd.to_datetime(px.index); px=px.sort_index()
r=px.pct_change()
base=((px.pct_change(20)-px.pct_change(60)/3)/(r.rolling(30).std()*np.sqrt(20)))
for sw in [3,5,10]:
 f=base.rolling(sw,min_periods=3).mean().shift(1)
 obs=[]; turns=[]
 for i in range(len(px)-10):
  x=f.iloc[i]; y=px.iloc[i+10]/px.iloc[i]-1
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: obs.append(z.corr(method='spearman').iloc[0,1])
  if i>0:
   a=f.iloc[i-1].rank(); b=x.rank(); q=pd.concat([a,b],axis=1).dropna()
   if len(q)>=8: turns.append((q.iloc[:,0]-q.iloc[:,1]).abs().mean()/max(len(q),1))
 a=np.array(obs); print('sw',sw,'dates',len(a),'instruments',px.shape[1],'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'turn',np.nanmean(turns),'coverage',f.notna().mean().mean())
