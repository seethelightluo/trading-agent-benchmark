import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-06-02'); rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];d['sig']=-(d.open/d.close.shift(1)-1);d['fwd']=d.close/d.open-1;d['symbol']=s;rows.append(d[['date','symbol','sig','fwd']])
x=pd.concat(rows);a=[];ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1:a.append(spearmanr(g.sig,g.fwd).statistic);ns.append(len(g))
a=np.array(a);print('dates',len(a),'rows',len(x.dropna()),'avg_names',np.mean(ns),'coverage',len(x.dropna())/(15*x.date.nunique()),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
for y,g in x.groupby(x.date.dt.year):
 q=[]
 for _,h in g.groupby('date'):
  h=h.dropna()
  if len(h)>=8 and h.sig.nunique()>1 and h.fwd.nunique()>1:q.append(spearmanr(h.sig,h.fwd).statistic)
 if q:print(y,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
x.dropna()[['date','symbol','sig']].to_csv('scripts/miner_1_20270603_overnight_signal.csv',index=False)
