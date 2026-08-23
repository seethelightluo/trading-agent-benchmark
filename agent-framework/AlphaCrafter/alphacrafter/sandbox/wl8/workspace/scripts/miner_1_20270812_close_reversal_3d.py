import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-08-11')
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 d['sig']=-d.close.pct_change(3); d['fwd']=d.close.shift(-1)/d.close-1; d['symbol']=s; rows.append(d[['date','symbol','sig','fwd']])
x=pd.concat(rows); v=x.dropna(); ics=[]; ns=[]
for dt,g in v.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1: ics.append(spearmanr(g.sig,g.fwd).statistic); ns.append(len(g))
a=np.array(ics); print('dates',len(a),'rows',len(v),'avg_names',round(np.mean(ns),2),'coverage',round(len(v)/(15*x.date.nunique()),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for y,g in v.groupby(v.date.dt.year):
 z=[]
 for _,h in g.groupby('date'):
  if len(h)>=8 and h.sig.nunique()>1 and h.fwd.nunique()>1:z.append(spearmanr(h.sig,h.fwd).statistic)
 if len(z)>1: print('regime',y,'dates',len(z),'IC',round(np.mean(z),6),'ICIR',round(np.mean(z)/np.std(z,ddof=1),6),'hit',round((np.array(z)>0).mean(),4))
for label,cut in [('recent90',END-pd.Timedelta(days=90)),('recent180',END-pd.Timedelta(days=180))]:
 z=[]
 for _,h in v[v.date>=cut].groupby('date'):
  if len(h)>=8 and h.sig.nunique()>1 and h.fwd.nunique()>1:z.append(spearmanr(h.sig,h.fwd).statistic)
 print(label,'dates',len(z),'IC',round(np.mean(z),6),'ICIR',round(np.mean(z)/np.std(z,ddof=1),6) if len(z)>1 else None)
v[['date','symbol','sig']].to_csv('scripts/miner_1_20270812_close_reversal_3d_signal.csv',index=False)
