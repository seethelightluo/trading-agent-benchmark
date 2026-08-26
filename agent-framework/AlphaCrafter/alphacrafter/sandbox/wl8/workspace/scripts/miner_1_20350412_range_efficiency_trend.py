import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d['close'].replace(0,np.nan)
p=pd.DataFrame(px).sort_index(); r=px and p.pct_change()
# Net directional movement divided by total path movement; multiply by signed return and volatility-normalize.
ret20=p.pct_change(20); path=r.abs().rolling(20,min_periods=15).sum(); eff=ret20.abs()/(path+1e-12)
vol=r.rolling(20,min_periods=15).std(); fac=np.sign(ret20)*eff/(vol+1e-5)
# lag to ensure only completed day; forward 10-day return
fwd=p.shift(-10)/p-1
rows=[]; sigrows=[]
for dt in fac.index:
 x=fac.loc[dt].shift if False else fac.loc[dt]
 y=fwd.loc[dt]
 z=pd.concat([x,y],axis=1).dropna(); z.columns=['f','y']
 if len(z)>=8:
  ic=spearmanr(z.f,z.y).statistic
  rows.append((dt,ic,len(z)))
for s in U:
 for dt in fac.index:
  if pd.notna(fac.loc[dt,s]): sigrows.append((dt,s,float(fac.loc[dt,s])))
ics=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# use last available historical date and report windows
print('dates',len(ics),'avg_n',ics.n.mean(),'coverage',len(sigrows)/(len(fac.index)*15))
print('ic',ics.ic.mean(),'icir',ics.ic.mean()/ics.ic.std(),'hit', (ics.ic>0).mean(),'turnover',fac.rank(pct=True).diff().abs().stack().mean())
for w in [365,750,1260]:
 q=ics.tail(w); print('window',w,'ic',q.ic.mean(),'icir',q.ic.mean()/q.ic.std(),'n',len(q))
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; rr=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr),len(rr))
ics.to_csv('scripts/miner_1_20350412_range_efficiency_trend_ic.csv')
pd.DataFrame(sigrows,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20350412_range_efficiency_trend_signal.csv',index=False)
