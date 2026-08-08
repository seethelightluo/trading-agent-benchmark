"""One idea: inverse persistent peer-relative overnight-gap pressure (20 sessions).
A repeated overnight gap versus the cross-asset median can reflect short-lived
price pressure rather than durable information.  The inverse signal tests for
mean reversion, scaled by each asset's gap variability.  All signals lag one
completed session."""
import runpy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
from alphacrafter.sim.utils import get_stock_daily_data
opens={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 opens[a]=pd.to_numeric(d.sort_values('date').set_index('date').open,errors='coerce')
O=pd.DataFrame(opens).reindex(P.index)
gap=O/P.shift(1)-1
relgap=gap.sub(gap.median(axis=1),axis=0)
gapvol=relgap.rolling(20,min_periods=15).std().replace(0,np.nan)
cand=(-relgap.rolling(20,min_periods=15).mean()/gapvol).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];y=fw[h].reindex(x.index); vals=[];breadth=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):vals.append(v);breadth.append(len(q))
 vals=np.asarray(vals)
 if len(vals)<2:return {'dates':len(vals)}
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR inverse_persistent_peer_relative_overnight_gap_pressure_20 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for name,period in [('2020_22',('2020-01-01','2022-12-31')),('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',name,stats(10,period))
mx=-1.;who='';evidence=0
for name,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 print('LIBCORR',name,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx:mx=float(abs(rho));who=name;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S))
