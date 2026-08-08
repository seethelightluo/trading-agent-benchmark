"""One idea: moderate-dispersion-conditioned peer-relative serial-dependence reversal."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); rel=R.sub(R.median(axis=1),axis=0)
# negative 20d lag-one autocorrelation; retain signal only when contemporaneous 20d peer dispersion is at/below its trailing 60d median.
def ac(x): return x.rolling(20,min_periods=16).corr(x.shift(1))
raw=-pd.DataFrame({a:ac(rel[a]) for a in A}); disp=rel.rolling(20,min_periods=16).std().median(axis=1)
quiet=disp<=disp.rolling(60,min_periods=40).median()
F=raw.where(quiet, np.nan).sub(raw.where(quiet,np.nan).median(axis=1),axis=0).shift(1)
def met(h,lo=None,hi=None):
 x=F.loc[lo:hi];y=(P.shift(-h)/P-1).reindex(x.index); z=[];ns=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 if not z:return {}
 z=np.array(z);return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),4),'mean_n':round(np.mean(ns),2),'min_n':min(ns)}
cut=P.index.max(); print('FACTOR quiet_dispersion_inverse_peer_relative_serial_dependence_20 CUTOFF',cut.date(),'ASSETS',len(A),'PRICE_DATES',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'COVERAGE',round(F.notna().stack().mean(),6),'TURNOVER',round(F.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(F.std(axis=1).mean(),6),'QUIET_DATES',int(quiet.sum()))
for h in (1,5,10,20):print('H',h,met(h))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]: print('REGIME20',n,met(20,lo,hi))
print('Novelty audit intentionally not run: only required before persistence; candidate must first pass IC/ICIR gate.')
