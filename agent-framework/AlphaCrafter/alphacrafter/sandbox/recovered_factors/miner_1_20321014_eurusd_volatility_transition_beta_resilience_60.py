"""One idea: EURUSD-volatility-transition beta resilience, 60 sessions.
Tests cross-asset relative EURUSD beta specifically when EURUSD absolute-return
volatility is elevated versus its own long baseline; all signals lag one session.
"""
import runpy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
e=get_index_daily_data('EURUSD',5000).copy(); e.date=pd.to_datetime(e.date)
er=pd.to_numeric(e.sort_values('date').set_index('date').close,errors='coerce').reindex(P.index).pct_change()
def beta(x,y,mask=None,w=60):
 q=pd.concat([x.rename('x'),y.rename('y')],axis=1)
 if mask is not None:q=q.where(mask,axis=0)
 return q.x.rolling(w,min_periods=12).cov(q.y)/q.y.rolling(w,min_periods=12).var()
# Elevated recent EURUSD movement, relative to a 60-session completed-history baseline.
state=er.abs().rolling(20,min_periods=15).mean()>er.abs().rolling(60,min_periods=40).mean()
raw=pd.DataFrame({a:beta(r[a],er,state)-beta(r[a],er) for a in A})
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index);v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   k=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(k):v.append(k);b.append(len(q))
 v=np.asarray(v)
 if len(v)<2:return {'dates':len(v)}
 return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR eurusd_volatility_transition_beta_resilience_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('STATE_SHARE',round(float(state.mean()),6),'CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stats(10,p))
mx=-1.;who='';evidence=0;invalid=[]
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if not np.isfinite(rho):invalid.append(n)
 elif abs(rho)>mx:mx=abs(rho);who=n;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S),'INVALID',invalid)
