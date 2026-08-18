import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index()
# Volatility-scaled lagged 3-day reversal: negative 3d return divided by trailing 20d realized vol.
ret=P.pct_change(); vol=ret.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(-ret.rolling(3).sum()/vol).shift(1)
print('assets',len(P.columns),'dates',len(P),'range',P.index.min(),P.index.max())
for h in [1,5,10]:
 fw=P.shift(-h)/P-1; ic=[]; n=[]; prev=None; turns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic.append(z.iloc[:,0].corr(z.iloc[:,1])); n.append(len(z)); r=f.loc[dt].rank(pct=True)
   if prev is not None: turns.append((r-prev).abs().mean())
   prev=r
 q=pd.Series(ic).dropna();
 print('horizon',h,'IC_dates',len(q),'avg_names',np.mean(n),'coverage',len(q)/max(1,len(P)-h),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean(),'turnover',np.mean(turns))
 for w in [120,252]:
  x=q.tail(w); print(' recent',w,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',(x>0).mean())
# signal artifact for reproducible downstream audit
out=f.tail(1).T.reset_index(); out.columns=['symbol','signal']; out.to_csv('scripts/miner_1_20341124_volscaled_reversal3_signal.csv',index=False)
