"""One idea: broad-stress-onset gated peer-relative reversal (5d return, 60d stress history)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']; C={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
p=pd.DataFrame(C); r=p.pct_change(); ret5=p.pct_change(5)
# Gate activates only at a genuinely broad 5d stress onset: current peer median is in its own trailing 60d lower quintile.
broad=ret5.median(axis=1); threshold=broad.rolling(60,min_periods=45).quantile(.20)
gate=(broad<=threshold).astype(float)
# During stress onset prefer the largest peer-relative losers (reversal); otherwise use neutral cross-sectional signal.
sig=(-ret5.sub(ret5.median(axis=1),axis=0)).mul(gate,axis=0).shift(1)
fwd={h:p.shift(-h).div(p).sub(1) for h in (1,5,10,20)}
def stat(h,lo=None,hi=None):
 x=sig.loc[lo:hi] if lo else sig; vals=[]; breadth=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
  # On inactive dates signal is identically zero and correctly has no ranking prediction.
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);breadth.append(len(q))
 vals=np.array(vals)
 return {'dates':len(vals),'ic':round(vals.mean(),6),'icir':round(vals.mean()/vals.std(ddof=1),6),'hit':round((vals>0).mean(),6),'breadth':round(np.mean(breadth),3),'min_breadth':min(breadth)}
cut=p.dropna(how='all').index.max(); rank=sig.rank(axis=1,pct=True)
print('FACTOR broad_stress_onset_peer_reversal_5_60 CUTOFF',cut.date(),'ASSETS',len(assets))
print('GATE_DATES',int(gate.sum()),'/',len(gate),'RATE',round(gate.mean(),6))
print('CELLS',int(sig.notna().sum().sum()),'/',sig.size,'COVERAGE',round(sig.notna().stack().mean(),6),'TURNOVER_ACTIVE',round(rank.diff().abs().stack().mean(),6))
for h in (1,5,10,20): print('H',h,stat(h))
for n,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cut.date())),('recent180',str(cut-pd.Timedelta(days=180)),str(cut.date()))]:print('REGIME10',n,stat(10,lo,hi))
# Pre-admission novelty proxies only; a complete library audit is required if predictive gates pass.
for n,x in {'simple_peer_reversal_5':-ret5.sub(ret5.median(axis=1),axis=0),'poststress_proxy':(-p.pct_change(10).sub(p.pct_change(10).median(axis=1),axis=0)).mul((broad<0).astype(float),axis=0),'risk_adjusted_trend':p.pct_change(20).div(r.rolling(20,min_periods=15).std())}.items():
 q=pd.concat([sig.stack(),x.shift(1).stack()],axis=1).dropna()
 print('PROXY',n,'cells',len(q),'rho',round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6))
