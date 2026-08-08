"""One candidate: USDCNY shock-beta resilience (60 sessions).
For each asset, measure its beta to USDCNY returns on the largest quartile of
absolute USDCNY moves, minus its unconditional beta.  A positive signal means
an asset's FX-shock response is stronger than its ordinary FX linkage.
USDCNY is observation-only and is never a candidate holding.  Signal is lagged
one complete session; all statistics use only returned visible history.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']
def ser_asset(a):
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
def ser_index(a):
 d=get_index_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame({a:ser_asset(a) for a in A}).sort_index(); R=P.pct_change()
fx=ser_index('USDCNY').reindex(P.index).ffill().pct_change()
# A trailing 60-session percentile is established at t from data through t.
threshold=fx.abs().rolling(60,min_periods=45).quantile(.75)
event=fx.abs().gt(threshold)
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for t in range(60,len(P)):
 idx=P.index[t-59:t+1]; x=fx.loc[idx]; ev=event.loc[idx]
 for a in A:
  y=R.loc[idx,a]; good=x.notna()&y.notna()
  base=good.sum(); shock=(good&ev).sum()
  if base>=45 and shock>=15 and x[good].var()>0 and x[good&ev].var()>0:
   F.loc[P.index[t],a]=y[good&ev].cov(x[good&ev])/x[good&ev].var()-y[good].cov(x[good])/x[good].var()
F=F.sub(F.median(axis=1),axis=0).shift(1);cut=P.index.max()
def metric(h,lo=None):
 x=F.loc[lo:] if lo else F;y=(P.shift(-h)/P-1).reindex(x.index); z=[];n=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);n.append(len(q))
 z=np.array(z);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(n)),2),'min_n':int(min(n))}
print('FACTOR usdcny_shock_beta_resilience_60 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20):print('H',h,metric(h))
for lab,lo in [('2020_22','2020-01-01'),('2023_24','2023-01-01'),('2025_26','2025-01-01'),('2027_current','2027-01-01'),('recent180',str(cut-pd.Timedelta(days=180)))]:print('REGIME10',lab,metric(10,lo))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
print('ADMISSION_CORRELATION pending: only compute if performance gates pass.')
