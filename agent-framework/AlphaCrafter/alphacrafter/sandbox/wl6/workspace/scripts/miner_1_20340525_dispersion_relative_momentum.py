import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def f(s):
 for g in (get_index_daily_data,get_stock_daily_data):
  try:
   x=g(s,days=4000)
   if x is not None and len(x): return x
  except: pass
S={}
for s in U:
 x=f(s)
 if x is not None: S[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(S).sort_index(); r=p.pct_change(); r20=p.pct_change(20); v=r.rolling(20).std()*np.sqrt(20)
d=r.std(axis=1).rolling(20).mean(); rel=(d/d.rolling(120).median()).clip(.5,2)
peer=r20.sub(r20.mean(axis=1),axis=0); sig=(peer/(v*np.sqrt(20))).mul(rel,axis='index').shift(1)
sig.to_csv('scripts/miner_1_20340525_dispersion_relative_momentum_signal.csv',index_label='date')
print('assets',p.shape[1],'rows',len(p),'valid',int(sig.notna().sum().sum()))
for h in [5,10,20,40]:
 fw=p.shift(-h)/p-1; z=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   c=q.iloc[:,0].corr(q.iloc[:,1])
   if np.isfinite(c): z.append(c)
 a=np.array(z); m=a.mean(); ir=m/a.std(ddof=1)*np.sqrt(len(a)); print('h',h,'dates',len(a),'avg_n',round((p.loc[sig.index].notna() & fw.notna()).sum(axis=1).loc[sig.index[-len(a):]].mean(),2) if len(a) else 0,'IC',round(m,8),'ICIR',round(ir,6),'hit',round((a>0).mean(),4))
print('turnover',sig.rank(pct=True).diff().abs().stack().mean())
