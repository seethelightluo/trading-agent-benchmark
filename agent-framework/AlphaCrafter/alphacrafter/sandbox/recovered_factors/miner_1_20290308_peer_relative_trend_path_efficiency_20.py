"""One price-only idea: peer-relative trend-path efficiency, 20 observations.
An asset's cumulative excess return versus the daily cross-asset median is divided
by the sum of its absolute daily excess moves.  A high value means the asset has
advanced versus peers along a direct rather than noisy path; unlike raw momentum,
the numerator is explicitly peer-relative and the denominator penalizes reversals.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; close={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 close[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(close); r=P.pct_change(); rel=r.sub(r.median(axis=1),axis=0)
# Require a nearly complete 20-session path; scale prevents near-zero denominator issues.
net=rel.rolling(20,min_periods=16).sum(); path=rel.abs().rolling(20,min_periods=16).sum()
f=(net/(path+1e-8)).shift(1); f=f.sub(f.median(axis=1),axis=0)
H=[1,5,10,20]; cutoff=P.dropna(how='all').index.max(); fw={h:P.shift(-h)/P-1 for h in H}
def ev(h, span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=fw[h].reindex(x.index); z=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v);ns.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z); return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR peer_relative_trend_path_efficiency_20 cutoff',cutoff.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2))
for h in H: print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
