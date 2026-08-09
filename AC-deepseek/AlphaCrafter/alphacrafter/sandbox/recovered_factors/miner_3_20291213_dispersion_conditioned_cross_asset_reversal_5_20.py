"""One candidate: dispersion-conditioned short-horizon cross-asset reversal.
The signal is the negative five-session return, activated proportionally when
20-session cross-asset return dispersion exceeds its own trailing median.
Motivation: heterogeneous, high-dispersion markets can make relative moves more
likely to mean-revert, while suppressing the signal in common-factor regimes.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
A=get_account_dict()['watch_list']
def close(a):
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame({a:close(a) for a in A}).sort_index(); R=P.pct_change()
# dispersion is observable at t; expanding only through prior day prevents look-ahead.
disp=R.std(axis=1).rolling(20,min_periods=15).mean()
base=disp.rolling(252,min_periods=126).median()
state=(disp/base).clip(lower=0.25,upper=3.0)
F=(-P.pct_change(5).mul(state,axis=0)).shift(1)
F=F.sub(F.median(axis=1),axis=0); cut=P.index.max()
def metric(h,lo=None):
 x=F.loc[lo:] if lo else F; y=(P.shift(-h)/P-1).reindex(x.index); z=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v); ns.append(len(q))
 z=np.array(z)
 return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR dispersion_conditioned_cross_asset_reversal_5_20 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20): print('H',h,metric(h))
for lab,lo in [('2020_22','2020-01-01'),('2023_24','2023-01-01'),('2025_26','2025-01-01'),('2027_current','2027-01-01'),('recent180',str(cut-pd.Timedelta(days=180)))]: print('REGIME10',lab,metric(10,lo))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
print('ADMISSION_CORRELATION pending: run only if IC/ICIR and multi-regime evidence pass.')
