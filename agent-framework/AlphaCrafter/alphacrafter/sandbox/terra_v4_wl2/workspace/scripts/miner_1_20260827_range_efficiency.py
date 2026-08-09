import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try:
        z=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
        D[s]=z.loc[:'2026-08-26']
    except Exception as e: print('missing',s,e)
# Range-efficiency trend: net 20d return divided by total absolute daily movement.
# Uses only close through t, predicts t to t+1 (and longer horizons).
def calc(k):
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change()
  f=r.rolling(20).sum()/(r.abs().rolling(20).sum()+1e-12)
  for i,dt in enumerate(x.index):
   if pd.notna(f.iloc[i]) and i+k<len(x):
    rows.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+k]/x.close.iloc[i]-1)))
 a=pd.DataFrame(rows,columns=['date','symbol','f','y']); ics=[]; names=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
   ics.append(spearmanr(g.f,g.y).statistic); names.append(len(g))
 q=np.asarray(ics); print('horizon',k,'dates',len(q),'avg_names',np.mean(names),'coverage',a.symbol.nunique()/15,'IC %.8f ICIR %.8f hit %.4f turnover_na'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0)))
 for yr in range(2020,2027):
  v=q[[pd.Timestamp(d).year==yr for d in a.groupby('date').size().index]] if False else None
 # calculate yearly directly
 for yr in range(2020,2027):
  vals=[v for dt,v in zip(names,[])];
  sub=[]
  for dt,g in a.groupby('date'):
   if dt.year==yr and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: sub.append(spearmanr(g.f,g.y).statistic)
  if sub: print(yr,'n',len(sub),'mean',round(np.mean(sub),5),'icir',round(np.mean(sub)/np.std(sub,ddof=1),4))
for k in (1,5,10): calc(k)
print('instruments',len(D),'date range',min(x.index.min() for x in D.values()),max(x.index.max() for x in D.values()))
