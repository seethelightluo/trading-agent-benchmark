"""One idea: drawdown-recovery convexity (60d drawdown, 10d rebound).
After an asset has suffered a substantial own 60-session drawdown, a recent
10-session rebound may signal a durable repair rather than raw short-horizon
momentum. The interaction preserves full cross-asset coverage and tests that
specific path-dependent continuation hypothesis.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C).sort_index(); R=P.pct_change()
# Strictly prior drawdown state prevents current rebound from changing its denominator.
prior=P.shift(10); peak=prior.rolling(60,min_periods=45).max(); dd=(prior/peak-1).clip(upper=0)
rebound=P/P.shift(10)-1
# Positive only when a rebound follows an already damaged path; median centering is cross-sectional.
f=(rebound*(-dd)).shift(1); f=f.sub(f.median(axis=1),axis=0)
H=[1,5,10,20]; FW={h:P.shift(-h)/P-1 for h in H}; cutoff=P.dropna(how='all').index.max()
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]; y=FW[h].reindex(x.index); z=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z); return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR drawdown_recovery_convexity_60_10 cutoff',cutoff.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2))
print('MEAN_PRIOR_DRAWDOWN',round(float(dd.mean().mean()),6),'MEAN_REBOUND',round(float(rebound.mean().mean()),6))
for h in H:print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
