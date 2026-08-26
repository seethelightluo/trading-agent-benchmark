import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-07-20')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
r=p.pct_change(); defs=['XAU','US10Y','CN10Y']
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
# relative leadership: medium-term return relative to defensive basket, lagged and cross-section centered
for look in [20,40,60]:
 x=p.shift(1)/p.shift(look+1)-1; f=x.sub(x[defs].mean(axis=1),axis=0)
 f=f.sub(f.median(axis=1),axis=0).rolling(3,min_periods=3).mean()
 print('\nLOOK',look)
 for h in [1,5,10,20]:
  vals=[]; ns=[]
  for i,d in enumerate(p.index):
   if d<pd.Timestamp('2020-03-01') or d>cut or i+h>=len(p):continue
   q=ic(f.loc[d],(p.shift(-h)/p-1).loc[d])
   if pd.notna(q):vals.append(q);ns.append((f.loc[d].notna()&((p.shift(-h)/p-1).loc[d].notna())).sum())
  z=pd.Series(vals); print('h',h,'dates',len(z),'n',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
 # regime blocks and recent
 vals=[]; ds=[]
 for i,d in enumerate(p.index):
  if d<pd.Timestamp('2020-03-01') or d>cut or i+10>=len(p):continue
  q=ic(f.loc[d],(p.shift(-10)/p-1).loc[d])
  if pd.notna(q):vals.append(q);ds.append(d)
 z=pd.Series(vals,index=ds)
 print('recent365',z.tail(365).mean(),z.tail(365).mean()/z.tail(365).std(ddof=1),'recent180',z.tail(180).mean(),z.tail(180).mean()/z.tail(180).std(ddof=1))
 if look==40:
  f.to_csv('scripts/miner_2_20330721_defensive_relative_leadership_signal.csv')
  pd.DataFrame({'date':z.index,'ic':z.values}).to_csv('scripts/miner_2_20330721_defensive_relative_leadership_ic.csv',index=False)
 print('coverage',f.loc[p.index[(p.index>=pd.Timestamp('2020-03-01'))&(p.index<=cut)]].notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
