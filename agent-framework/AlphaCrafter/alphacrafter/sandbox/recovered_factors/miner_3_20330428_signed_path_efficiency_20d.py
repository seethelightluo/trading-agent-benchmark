"""Miner_3 single-idea validation: cross-asset signed path efficiency, 20 days.
High efficiency means a persistent directional path (net 20d return divided by cumulative
absolute daily moves), tested as a continuation signal with completed daily bars only.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-04-27')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change()
# A unitless directional-efficiency signal; +1 is a steady rise and -1 a steady decline.
f=(p.pct_change(20).div(r.abs().rolling(20,min_periods=14).sum())).replace([np.inf,-np.inf],np.nan)
print('CANDIDATE signed_path_efficiency_20d cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',int(f.notna().any(axis=1).sum()),'valid_cells',int(f.notna().sum().sum()),'coverage',round(float(f.notna().mean().mean()),6))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; vals=[]; ns=[]
 for d in f.index:
  q=pd.concat((f.loc[d].rename('f'),fw.loc[d].rename('y')),axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z));ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float); ics[h]=s; sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for nm,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=s.loc[lo:hi]; print('REGIME10',nm,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(ts)),6),'pairs',len(ts))
print('DECAY',{h:(round(float(x.mean()),6),round(float(x.mean()/x.std(ddof=1)),6),len(x)) for h,x in ics.items()})
f.to_pickle('scripts/miner_3_20330428_signed_path_efficiency_20d_signal.pkl')
