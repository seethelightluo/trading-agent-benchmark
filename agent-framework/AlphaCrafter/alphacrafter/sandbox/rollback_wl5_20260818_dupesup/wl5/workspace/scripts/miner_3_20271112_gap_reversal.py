import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 except Exception as e: print('missing',s,e)

def evaluate(name,fn):
 rows=[]
 for s,x in D.items():
  f=fn(x); y=x.close.shift(-1)/x.close-1
  for d in x.index:
   if d<pd.Timestamp('2020-01-02') or pd.isna(f.loc[d]) or pd.isna(y.loc[d]):continue
   rows.append((d,s,float(f.loc[d]),float(y.loc[d])))
 a=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
 for label,lo,hi in [('all','2020-01-01','2027-11-12'),('online','2026-07-16','2027-11-12'),('recent','2027-01-01','2027-11-12')]:
  q=a[(a.date>=lo)&(a.date<=hi)]; vals=[]; ns=[]
  for d,g in q.groupby('date'):
   if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1:
    vals.append(spearmanr(g.factor,g.fwd).statistic);ns.append(len(g))
  z=np.array(vals); ic=np.nanmean(z) if len(z) else np.nan; ir=ic/np.nanstd(z,ddof=1) if len(z)>1 else np.nan
  print(name,label,'dates',len(z),'avgN',round(np.mean(ns),2) if ns else 0,'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(np.mean(z>0),4) if len(z) else 0,'coverage',round(q.symbol.nunique()/15,4))

evaluate('overnight_gap_reversal',lambda x: -(x.open/x.close.shift(1)-1))
evaluate('intraday_reversal',lambda x: -(x.close/x.open-1))
evaluate('clv',lambda x: (2*(x.close-x.low)/(x.high-x.low).replace(0,np.nan)-1))
