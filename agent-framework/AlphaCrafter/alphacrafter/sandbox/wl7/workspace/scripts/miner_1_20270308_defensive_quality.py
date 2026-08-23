import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f): px[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change();
# Defensive quality: lagged inverse realized volatility, conditioned on positive medium trend
v=r.shift(1).rolling(20).std(); trend=p.shift(1).pct_change(60)
f=1/v * (1+0.5*np.tanh(trend/0.10))
rows=[]; art=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  for s,x in f.loc[dt].items(): art.append((dt,s,x))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=a[a.index>=pd.Timestamp('2025-01-01')]
print('dates',len(q),'all_dates',len(a),'avg_n',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252),'hit',(q.ic>0).mean(),'coverage',q.n.mean()/15)
print('early_late',q.iloc[:len(q)//2].ic.mean(),q.iloc[len(q)//2:].ic.mean())
for h in [1,5,10,20]:
 yy=p.shift(-h).pct_change(h); zlist=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:zlist.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(zlist),len(zlist))
pd.DataFrame(art,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20270308_defensive_quality_signal.csv',index=False)
