import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-07-14'); frames=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 r=d.close.pct_change(); down=r.where(r<0,0.0)
 d['sig']=-d.close.pct_change(5)/(down.rolling(20,min_periods=15).std()*np.sqrt(5))
 d['fwd']=d.close.shift(-1)/d.close-1; d['symbol']=s
 frames.append(d[['date','symbol','sig','fwd']])
x=pd.concat(frames,ignore_index=True); valid=x.dropna(subset=['sig','fwd']); ics=[]; ns=[]
for dt,g in valid.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1: ics.append(spearmanr(g.sig,g.fwd).statistic); ns.append(len(g))
a=np.asarray(ics)
print('dates',len(a),'rows',len(valid),'avg_names',round(np.mean(ns),2),'coverage',round(len(valid)/(15*x.date.nunique()),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for h in [1,5,10]:
 out=[]
 for s in U:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; r=d.close.pct_change(); down=r.where(r<0,0.0); sig=-d.close.pct_change(5)/(down.rolling(20,min_periods=15).std()*np.sqrt(5)); f=d.close.shift(-h)/d.close-1
  frames2=pd.DataFrame({'date':d.date,'sig':sig,'f':f,'symbol':s}); out.append(frames2)
 q=pd.concat(out).dropna(); z=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.f.nunique()>1:z.append(spearmanr(g.sig,g.f).statistic)
 z=np.asarray(z); print('horizon',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
for y,g in valid.groupby(valid.date.dt.year):
 z=[]
 for _,h in g.groupby('date'):
  if len(h)>=8 and h.sig.nunique()>1 and h.fwd.nunique()>1:z.append(spearmanr(h.sig,h.fwd).statistic)
 if len(z)>1:print('regime',y,'dates',len(z),'IC',round(np.mean(z),6),'ICIR',round(np.mean(z)/np.std(z,ddof=1),6))
valid[['date','symbol','sig']].to_csv('scripts/miner_1_20270729_downside_adjusted_reversal_signal.csv',index=False)
