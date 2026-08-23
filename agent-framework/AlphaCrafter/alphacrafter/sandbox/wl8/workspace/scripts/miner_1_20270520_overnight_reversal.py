import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-05-19')
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x['date']=pd.to_datetime(x['date']); x=x[x.date<=END].sort_values('date').set_index('date')
 # Contrarian overnight gap from prior close to today's open, predicting today's close return.
 x['sig']=-(x['open']/x['close'].shift(1)-1); x['fwd']=x['close']/x['open']-1
 x=x.dropna(subset=['sig','fwd']); x['symbol']=s; rows.append(x.reset_index()[['date','symbol','sig','fwd']])
z=pd.concat(rows); ics=[]; nms=[]
for dt,g in z.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1: ics.append(spearmanr(g.sig,g.fwd).statistic); nms.append(len(g))
a=np.array(ics)
print('period',z.date.min().date(),z.date.max().date(),'dates',len(a),'rows',len(z),'avg_names',round(np.mean(nms),3),'coverage',round(len(z)/(len(U)*z.date.nunique()),5),'IC',round(np.nanmean(a),8),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),8),'hit',round(np.mean(a>0),5))
for y,g in z.groupby(z.date.dt.year):
 q=[]
 for _,h in g.groupby('date'):
  if len(h)>=8 and h.sig.nunique()>1 and h.fwd.nunique()>1:q.append(spearmanr(h.sig,h.fwd).statistic)
 if q: print(y,len(q),round(np.mean(q),5),round(np.mean(q)/np.std(q,ddof=1),5))
for h in [1,5]:
 rr=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date'); x['sig']=-(x.open/x.close.shift(1)-1); x['fwd']=x.close.shift(-h)/x.close-1; rr.append(x[['date','sig','fwd']].assign(symbol=s))
 q=pd.concat(rr).dropna(); vals=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1: vals.append(spearmanr(g.sig,g.fwd).statistic)
 print('horizon',h,'dates',len(vals),'IC',round(np.mean(vals),8),'ICIR',round(np.mean(vals)/np.std(vals,ddof=1),8))
z[['date','symbol','sig']].sort_values(['date','symbol']).to_csv('scripts/miner_1_20270520_overnight_reversal_signal.csv',index=False)
