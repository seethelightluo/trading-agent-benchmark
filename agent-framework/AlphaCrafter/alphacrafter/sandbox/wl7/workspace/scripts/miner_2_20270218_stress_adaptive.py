import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s,n=2600):
 for fn in [get_stock_daily_data,get_index_daily_data]:
  try:d=fn(s,n)
  except Exception: continue
  if d is not None:
   d=d.copy(); d['date']=pd.to_datetime(d['date']); return d.set_index('date')['close'].astype(float)
 return None
px={s:get(s) for s in U}; px={s:x for s,x in px.items() if x is not None}; p=pd.DataFrame(px).sort_index().ffill(); v=pd.Series(20.,index=p.index)
r10=p.pct_change(10); vol=p.pct_change().rolling(20).std()*np.sqrt(20); stress=np.tanh(v.pct_change(5).rolling(20).mean()*8)
adj=pd.DataFrame(np.repeat((1-1.25*stress).shift(1).to_numpy()[:,None],len(p.columns),axis=1),index=p.index,columns=p.columns)
sig=(r10/vol)*adj; rows=[]
for i in range(len(p)-20):
 a=sig.iloc[i]
 for h in [1,5,10,20]:
  y=p.iloc[i+h]/p.iloc[i]-1; z=pd.concat([a,y],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0: rows.append((p.index[i],h,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
df=pd.DataFrame(rows,columns=['date','h','n','ic']); print('candidate stress_adaptive_vol_adjusted_10d_momentum dates',df.date.nunique(),'assets',len(px),'coverage',sig.notna().mean().mean())
for h in [1,5,10,20]:
 q=df[df.h==h].ic; print(h,'dates',len(q),'avg_n',df[df.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std()*np.sqrt(252),'hit',(q>0).mean())
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean());out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270218_stress_adaptive_signal.csv',index=False)
