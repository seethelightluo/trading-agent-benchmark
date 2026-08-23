import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; data={}
for s in watch:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);data[s]=d.set_index('date').sort_index()
rows=[]
for s,d in data.items():
 r=d.close.pct_change(); mom=d.close/d.close.shift(20)-1
 down=np.sqrt((r.clip(upper=0)**2).rolling(20).mean())
 sig=(mom/(down+1e-8)).shift(1); fwd=d.close.shift(-10)/d.close-1
 rows.append(pd.DataFrame({'date':d.index,'s':s,'sig':sig.values,'fwd':fwd.values}).dropna())
x=pd.concat(rows); vals=[]
for dt,g in x.groupby('date'):
 if len(g)>=8: vals.append((dt,spearmanr(g.sig,g.fwd).statistic,len(g)))
a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date');print('dates',len(a),'avg_n',a.n.mean(),'coverage',len(x)/sum(len(d) for d in data.values()));print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=a.loc[lo:hi].ic;print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [1,5,10,20]:
 z=[]
 for s,d in data.items():
  r=d.close.pct_change(); mom=d.close/d.close.shift(20)-1; down=np.sqrt((r.clip(upper=0)**2).rolling(20).mean());sig=(mom/(down+1e-8)).shift(1);z.append(pd.DataFrame({'date':d.index,'sig':sig.values,'fwd':(d.close.shift(-h)/d.close-1).values}).dropna())
 xx=pd.concat(z);ii=[]
 for dt,g in xx.groupby('date'):
  if len(g)>=8:ii.append(spearmanr(g.sig,g.fwd).statistic)
 print('decay',h,np.nanmean(ii),len(ii))
x.pivot(index='date',columns='s',values='sig').to_csv('scripts/miner_1_20270615_downside_sortino_signal.csv')
