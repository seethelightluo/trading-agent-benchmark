import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,5000)
   if x is not None and len(x)>100:return x
  except: pass
D={s:load(s) for s in U};D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items()}).sort_index().groupby(level=0).last();R=C.ffill().pct_change();res=R.sub(R.mean(axis=1),axis=0)
shock=res.rolling(20,min_periods=15).sum().shift(1); vol=R.rolling(40,min_periods=25).std().shift(1)
breadth=(R<0).mean(axis=1).shift(1); macro=R.mean(axis=1).rolling(30,min_periods=20).mean().shift(1)
gate=((breadth>=.45)&(breadth<=.75)&(macro<0)).astype(float)
f=(-shock/vol).mul(gate,axis=0).replace([np.inf,-np.inf],np.nan)
print('assets',len(D),'dates',len(C),'active',int(gate.sum()),'coverage',round(f.notna().sum(axis=1).replace(0,np.nan).mean()/len(D),4))
for h in [1,3,5,10]:
 fr=R.rolling(h).sum().shift(-h);a=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
   if pd.notna(q):a.append(q);ds.append(d)
 a=pd.Series(a,index=ds);print('H',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
 if h==1:
  k=max(1,len(a)//2);print('regimes',round(a.iloc[:k].mean(),6),round(a.iloc[k:].mean(),6))
f.to_csv('scripts/miner_1_20330217_medium_shock20_signal.csv',index_label='date')
