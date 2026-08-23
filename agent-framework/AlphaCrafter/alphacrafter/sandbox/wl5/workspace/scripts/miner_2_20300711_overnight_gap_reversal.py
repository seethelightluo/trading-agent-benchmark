import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-07-10')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; d={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); d[a]=x.sort_values('date').set_index('date')
def make(h):
 rows=[]
 for a,x in d.items():
  gap=-(x.open/x.close.shift(1)-1); fwd=x.close.shift(-h)/x.close-1
  rows.append(pd.DataFrame({'date':x.index,'asset':a,'factor':gap.values,'fwd':fwd.values}))
 return pd.concat(rows,ignore_index=True).query('date <= @CUT')
def calc(q):
 vals=[]; ns=[]
 for dt,g in q.replace([np.inf,-np.inf],np.nan).dropna().groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1:
   z=spearmanr(g.factor,g.fwd).statistic
   if np.isfinite(z): vals.append(z); ns.append(len(g))
 v=np.array(vals); return len(v),np.mean(ns),v.mean(),v.mean()/v.std(ddof=1),np.mean(v>0)
r=make(5); print('cutoff',CUT.date(),'dates/assets/coverage/IC/ICIR/hit',calc(r),r.dropna().shape[0]/len(r))
for lo,hi in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2030-07-10')]: print('regime',lo,hi,calc(r[(r.date>=lo)&(r.date<=hi)]))
for h in [1,3,5,10,20]: print('horizon',h,calc(make(h)))
r.to_csv('scripts/miner_2_20300711_overnight_gap_reversal_signal.csv',index=False)
