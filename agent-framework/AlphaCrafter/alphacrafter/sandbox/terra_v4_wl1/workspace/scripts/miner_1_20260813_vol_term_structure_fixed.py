import pandas as pd,numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-07-15')
# Per-asset calendars: forward returns use next h observed closes, avoiding exact-date alignment loss.
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date')
 close=pd.to_numeric(d.close,errors='coerce'); ret=close.pct_change()
 v20=ret.rolling(20,min_periods=15).std(); v120=ret.rolling(120,min_periods=80).std()
 fac=-(v20/v120).replace([np.inf,-np.inf],np.nan)
 for h in (1,5,10):
  y=close.shift(-h)/close-1
  x=pd.DataFrame({'date':close.index,'asset':s,'factor':fac.values,'fwd':y.values,'h':h}).dropna()
  rows.append(x)
z=pd.concat(rows,ignore_index=True)
def stats(q):
 vals=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>=3 and g.fwd.nunique()>=3:
   vals.append(spearmanr(g.factor,g.fwd).statistic); ns.append(len(g))
 a=pd.Series(vals,dtype=float); return len(a),float(np.mean(ns)) if ns else np.nan,float(a.mean()) if len(a) else np.nan,float(a.mean()/a.std(ddof=1)*np.sqrt(252)) if len(a)>1 else np.nan,float((a>0).mean()) if len(a) else np.nan
for h in (1,5,10): print('H',h,'N avgN IC ICIR hit',stats(z[z.h==h]))
q=z[z.h==1]
# regime observations and coverage
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]: print('regime',lo,hi,stats(q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]))
# rank turnover on adjacent observations per asset
ranks=z[z.h==1].pivot(index='date',columns='asset',values='factor').rank(axis=1,pct=True)
print('turnover',float(ranks.diff().abs().mean(axis=1).mean()),'coverage',float(q.groupby('date').size().loc[lambda x:x>=8].mean()/15),'raw rows',len(q),'cutoff',cutoff.date())
