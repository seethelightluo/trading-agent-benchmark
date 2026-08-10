import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x
  except: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change()
# Candidate: medium-term relative strength, market-neutralized by contemporaneous cross-sectional median.
# Positive values mean asset outperformed peers over prior 20 sessions.
raw=px.pct_change(20)
f=raw.sub(raw.median(axis=1),axis=0)
for h in [1,5,10]:
 fr=px.shift(-h).div(px)-1
 vals=[]; dates=[]; ns=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8:
   vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(d); ns.append(len(a))
 z=pd.Series(vals,index=pd.to_datetime(dates)).dropna()
 print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'std',round(z.std(),6))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-12-31')]:
  q=z.loc[lo:hi]
  if len(q): print('REG',lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
# daily value and date coverage
print('coverage',round(f.notna().sum().sum()/(f.shape[0]*len(U)),4),'date coverage',round(f.notna().any(axis=1).mean(),4),'period',px.index.min(),px.index.max())
# artifact for audit
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_relstrength20_resid.csv',index=False)
