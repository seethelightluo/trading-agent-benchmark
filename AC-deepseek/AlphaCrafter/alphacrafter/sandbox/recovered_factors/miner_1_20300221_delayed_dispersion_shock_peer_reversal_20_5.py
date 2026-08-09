"""One candidate: delayed dispersion-shock peer reversal (20/5).
A dispersion shock identifies unusually heterogeneous cross-asset returns; the
signal reverses each asset's relative five-day move ending five sessions before
the decision, avoiding the adverse immediate post-shock reversal direction.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def px(a):
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame({a:px(a) for a in A}).sort_index(); R=P.pct_change(); cutoff=P.index.max()
# At t, only the completed values ending t-1 are used after the final shift.
rel=P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0)
disp=R.std(axis=1)
shock=(disp>disp.rolling(60,min_periods=40).quantile(.80)).astype(float)
# Use a 20-session event average and an extra five-session delay; zero event
# count has no usable signal rather than manufacturing a neutral ranking.
num=(-rel.shift(5)).mul(shock.shift(5),axis=0).rolling(20,min_periods=5).sum()
den=shock.shift(5).rolling(20,min_periods=5).sum()
F=num.div(den,axis=0).where(den>0).shift(1)
F=F.sub(F.median(axis=1),axis=0)
def met(h,lo=None):
 x=F.loc[lo:] if lo else F; y=(P.shift(-h)/P-1).reindex(x.index); z=[]; n=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);n.append(len(q))
 z=np.array(z)
 return dict(dates=len(z),ic=round(float(z.mean()),6),icir=round(float(z.mean()/z.std(ddof=1)),6),hit=round(float((z>0).mean()),4),mean_n=round(float(np.mean(n)),2),min_n=int(min(n)))
print('FACTOR delayed_dispersion_shock_peer_reversal_20_5 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'COVERAGE',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20):print('H',h,met(h))
for lab,lo in [('2020_22','2020-01-01'),('2023_24','2023-01-01'),('2025_26','2025-01-01'),('2027_now','2027-01-01'),('recent180',str(cut-pd.Timedelta(days=180)))]:print('REGIME10',lab,met(10,lo))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
print('LIBRARY_CORRELATION_PENDING only if performance admission passes')
