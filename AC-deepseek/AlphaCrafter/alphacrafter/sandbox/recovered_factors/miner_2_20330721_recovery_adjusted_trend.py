import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); px[a]=d.set_index('date').close
p=pd.DataFrame(px).sort_index(); ret=p.pct_change()
# Novel candidate: recovery-adjusted medium-term trend. Favor positive 60d trend,
# but penalize assets whose current close is far below its 60d peak; lag one day.
trend=p.pct_change(60)
dd=p/p.rolling(60).max()-1
sig=(trend/(1+(-dd).clip(lower=0))).shift(1)
print('range',p.index.min(),p.index.max(),'assets',len(px),'rows',len(p))
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   q=spearmanr(sig.loc[dt,ok],fwd.loc[dt,ok]).statistic
   if np.isfinite(q): vals.append(q); ds.append(dt); ns.append(ok.sum())
 z=np.asarray(vals); print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/(z.std(ddof=1)+1e-12),np.mean(z>0)))
 for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
  q=z[(np.asarray(ds)>=pd.Timestamp(lo+'-01-01'))&(np.asarray(ds)<=pd.Timestamp(hi+'-12-31'))]
  print(' ',lo+'-'+hi,len(q),('IC %.6f ICIR %.6f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12))) if len(q) else '')
ranks=sig.rank(axis=1,pct=True); print('turn10',ranks.diff(10).abs().mean(axis=1).dropna().mean(),'coverage',sig.notna().sum().sum()/sig.size)
