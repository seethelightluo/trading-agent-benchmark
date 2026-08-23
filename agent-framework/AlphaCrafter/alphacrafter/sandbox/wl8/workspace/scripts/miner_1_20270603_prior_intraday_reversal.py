import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-06-02'); rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=END].copy()
 # Prior session's open-to-close move, reversed for next close-to-close return.
 d['sig']=-(d.close/d.open-1).shift(1)
 d['fwd']=d.close.shift(-1)/d.close-1
 d['symbol']=s; rows.append(d[['date','symbol','sig','fwd']])
x=pd.concat(rows); vals=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1:
  vals.append(spearmanr(g.sig,g.fwd).statistic); ns.append(len(g))
a=np.array(vals)
print('period',x.date.min().date(),x.date.max().date(),'dates',len(a),'rows',len(x.dropna()),'avg_names',round(np.mean(ns),3),'coverage',round(len(x.dropna())/(15*x.date.nunique()),5),'IC',round(np.nanmean(a),8),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),8),'hit',round(np.mean(a>0),5))
for y,g in x.groupby(x.date.dt.year):
 q=[]
 for _,h in g.groupby('date'):
  h=h.dropna()
  if len(h)>=8 and h.sig.nunique()>1 and h.fwd.nunique()>1:q.append(spearmanr(h.sig,h.fwd).statistic)
 if q: print(y,len(q),round(np.mean(q),6),round(np.mean(q)/np.std(q,ddof=1),6))
for h in [1,5,10]:
 rr=[]
 for s in U:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END].copy()
  d['sig']=-(d.close/d.open-1).shift(1);d['fwd']=d.close.shift(-h)/d.close-1;d['symbol']=s;rr.append(d[['date','symbol','sig','fwd']])
 q=pd.concat(rr); vs=[]
 for _,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1:vs.append(spearmanr(g.sig,g.fwd).statistic)
 print('horizon',h,'dates',len(vs),'IC',round(np.mean(vs),8),'ICIR',round(np.mean(vs)/np.std(vs,ddof=1),8))
x.dropna()[['date','symbol','sig']].sort_values(['date','symbol']).to_csv('scripts/miner_1_20270603_prior_intraday_reversal_signal.csv',index=False)
