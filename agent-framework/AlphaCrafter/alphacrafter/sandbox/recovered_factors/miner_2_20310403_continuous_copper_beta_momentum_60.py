"""One candidate: continuous copper beta momentum (60 sessions).
A cross-asset cyclicality exposure factor: rolling beta of each asset's daily
returns to COPPER returns.  Unlike conditional upside beta it uses every
session, yielding dense estimates. Signal is lagged one completed day.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); cutoff=P.index.max()
# beta_i,COPPER = cov_60(r_i,r_copper)/var_60(r_copper), contemporaneous inputs only.
c=R['COPPER']; cv=c.rolling(60,min_periods=45).var().replace(0,np.nan)
B=R.rolling(60,min_periods=45).cov(c).div(cv,axis=0)
F=B.sub(B.median(axis=1),axis=0).shift(1)
def met(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); vals=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z); ns.append(len(q))
 if not vals:return {'dates':0}
 z=np.array(vals); sd=z.std(ddof=1)
 return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/sd),6) if sd else None,'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR continuous_copper_beta_momentum_60 cutoff',cutoff.date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20): print('H',h,met(h))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cutoff-pd.Timedelta(days=180)),None)]: print('REGIME10',n,met(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTION_SD',round(float(F.std(axis=1).mean()),6))
print('LIBRARY_CORRELATION not computed unless numeric/recent validation supports admission; missing complete evidence fails admission.')
