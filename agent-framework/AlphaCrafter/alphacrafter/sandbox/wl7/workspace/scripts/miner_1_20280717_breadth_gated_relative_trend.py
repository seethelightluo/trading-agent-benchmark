import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   q=fn(s,3000)
   if q is not None and len(q): d=q; break
  except Exception: pass
 if d is not None:
  D[s]=d[['date','close']].assign(date=lambda x:pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Candidate: regime-conditioned relative trend. 20d vol-scaled return relative to
# cross-sectional median, reversed when broad market trend is negative; lagged 1 day.
raw=r.rolling(20,min_periods=15).sum(); vol=r.rolling(40,min_periods=25).std()
breadth=(raw>0).mean(axis=1)
regime=np.where(breadth<0.40,-1.0,1.0)
sig=(raw.sub(raw.median(axis=1),axis=0)/(vol*np.sqrt(20)+1e-8)*regime[:,None]).shift(1).rank(axis=1,pct=True)
for h in (1,5,10,20):
 f=p.shift(-h)/p-1; vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(x): vals.append(x);ds.append(dt);ns.append(len(z))
 a=np.array(vals); ic=a.mean(); ir=ic/a.std(ddof=1)*np.sqrt(len(a))
 print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((a>0).mean(),4))
 if h==10:
  print('coverage',round(sig.notna().sum().sum()/(sig.shape[0]*len(U)),4),'turnover',round(sig.diff().abs().mean().mean(),6),'assets',len(D))
  for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027-28','2027','2028-12-31')]:
   q=a[[pd.Timestamp(lo)<=x<=pd.Timestamp(hi) for x in ds]]
   print('REG',lab,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),4) if len(q)>1 else None)
sig.to_csv('scripts/miner_1_20280717_breadth_gated_relative_trend_signal.csv',index_label='date')
print('range',p.index.min(),p.index.max(),'assets',len(D))
