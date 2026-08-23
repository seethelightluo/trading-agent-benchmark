import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not __import__('os').path.exists(f): f='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); d[s]=x.set_index('date').close
p=pd.DataFrame(d).sort_index(); r=p.pct_change()
# Novel interpretable candidate: relative trend persistence, 10d return versus peer median,
# scaled by own 20d vol, lagged one day; intended to distinguish idiosyncratic trend from beta.
cs=r.rolling(10).sum(); peer=cs.sub(cs.median(axis=1),axis=0)
vol=r.rolling(20).std()*np.sqrt(20)
f=(peer/vol).shift(1)
fr=p.shift(-1)/p-1
rows=[]
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.sum()/(len(x)*15))
print('IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit', (x.ic>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=x.loc[a:b].ic; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10,20]:
 fh=p.shift(-h)/p-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fh.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr),len(rr))
# artifact for provenance
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20270423_relative_trend_persistence_signal.csv',index=False)
