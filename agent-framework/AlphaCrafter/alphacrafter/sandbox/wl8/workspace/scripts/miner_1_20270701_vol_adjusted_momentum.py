import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-06-30'); rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=END].copy(); r=d.close.pct_change()
 # signal known at t close: medium momentum scaled by recent realized volatility
 d['sig']=d.close.pct_change(10)/(r.rolling(20,min_periods=15).std()*np.sqrt(10))
 d['fwd']=d.close.shift(-1)/d.close-1; d['symbol']=s
 rows.append(d[['date','symbol','sig','fwd']])
x=pd.concat(rows,ignore_index=True); valid=x.dropna(subset=['sig','fwd']); ics=[]; ns=[]; dates=[]
for dt,g in valid.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1:
  ics.append(spearmanr(g.sig,g.fwd).statistic); ns.append(len(g)); dates.append(dt)
a=np.asarray(ics)
print('dates',len(a),'rows',len(valid),'avg_names',round(np.mean(ns),2),'coverage',round(len(valid)/(15*x.date.nunique()),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for h in [1,5,10]:
 z=[]
 for s in U:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; r=d.close.pct_change(); sig=d.close.pct_change(10)/(r.rolling(20,min_periods=15).std()*np.sqrt(10)); f=d.close.shift(-h)/d.close-1
  z.append(pd.DataFrame({'date':d.date,'sig':sig,'f':f,'symbol':s}))
 q=pd.concat(z).dropna(); out=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.f.nunique()>1: out.append(spearmanr(g.sig,g.f).statistic)
 out=np.asarray(out);print('horizon',h,'dates',len(out),'IC',round(out.mean(),6),'ICIR',round(out.mean()/out.std(ddof=1),6))
for y,g in valid.groupby(valid.date.dt.year):
 q=[]
 for _,h in g.groupby('date'):
  if len(h)>=8 and h.sig.nunique()>1 and h.fwd.nunique()>1:q.append(spearmanr(h.sig,h.fwd).statistic)
 if len(q)>1: print('regime',y,'dates',len(q),'IC',round(np.mean(q),6),'ICIR',round(np.mean(q)/np.std(q,ddof=1),6))
valid[['date','symbol','sig']].to_csv('scripts/miner_1_20270701_vol_adjusted_momentum_signal.csv',index=False)
