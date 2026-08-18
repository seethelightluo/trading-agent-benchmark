import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Downside-risk normalized lagged 5-day reversal: recent loss is stronger when downside volatility is low.
down=r.where(r<0).rolling(20,min_periods=10).std()*np.sqrt(20)
f=(-r.rolling(5).sum()/down.replace(0,np.nan)).shift(1)
print('assets',len(P.columns),'dates',len(P),'range',P.index.min(),P.index.max())
for h in [1,5,10]:
 fw=P.shift(-h)/P-1; ic=[]; ns=[]; turns=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c): ic.append(c); ns.append(len(z))
   rr=f.loc[dt].rank(pct=True)
   if prev is not None: turns.append((rr-prev).abs().mean())
   prev=rr
 q=pd.Series(ic); print('horizon',h,'IC_dates',len(q),'avg_names',round(np.mean(ns),2),'coverage',round(len(q)/max(1,len(P)-h),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),4),'hit',round((q>0).mean(),4),'turnover',round(np.mean(turns),4))
 for w in [120,252]:
  x=q.tail(w); print(' recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),4),'hit',round((x>0).mean(),4))
out=f.tail(1).T.reset_index(); out.columns=['symbol','signal']; out.to_csv('scripts/miner_1_20341208_downside_norm_reversal5_signal.csv',index=False)
