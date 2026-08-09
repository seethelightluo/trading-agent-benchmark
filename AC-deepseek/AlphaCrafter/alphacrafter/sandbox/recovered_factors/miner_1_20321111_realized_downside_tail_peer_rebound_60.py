"""One idea: realized peer-relative rebound after idiosyncratic downside-tail events."""
import runpy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
# At session t, measure the asset's peer-relative return one completed session after
# its own lower-tail return.  The event cutoff is lagged, avoiding self-selection.
rel=r.sub(r.median(axis=1),axis=0)
event=pd.DataFrame({a:r[a] < r[a].rolling(60,min_periods=40).quantile(.20).shift(1) for a in A})
# Recent realized rebound conditional on a prior completed downside-tail event.
raw=rel.where(event.shift(1),axis=0).rolling(60,min_periods=12).mean()
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]; y=fw[h].reindex(x.index); vals=[];bs=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);bs.append(len(q))
 vals=np.array(vals)
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(bs)),3),'min_breadth':int(min(bs))} if len(vals)>1 else {'dates':len(vals)}
print('FACTOR realized_downside_tail_peer_rebound_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stats(10,p))
mx=-1;who='';ev=0;invalid=[]
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if not np.isfinite(rho):invalid.append(n)
 elif abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S),'INVALID',invalid)
