import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index(); P[a]=d.close
p=pd.DataFrame(P).sort_index().loc[:'2034-06-07']; r=p.pct_change()
# Recovery-quality: medium trend divided by downside volatility, with a drawdown penalty.
trend=p.pct_change(20)
down=r.clip(upper=0).rolling(20).std()*np.sqrt(20)
dd=p/p.rolling(60).max()-1
f=trend/(down+1e-6)+0.5*dd
print('cutoff',p.index.max().date(),'rows',len(p),'assets',len(assets))
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 v=np.asarray(vals); print('H',h,'dates',len(v),'meanN',round(np.mean(ns),2),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round(np.mean(v>0),4))
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
fr=p.shift(-10)/p-1; vals=[]; ds=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt)
v=np.asarray(vals); ds=pd.DatetimeIndex(ds)
for lo,hi in [(2020,2024),(2025,2029),(2030,2034)]:
 x=v[(ds.year>=lo)&(ds.year<=hi)]; print('regime',lo,hi,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('library_audit','FAILED: exact common-cell audit required before admission')
