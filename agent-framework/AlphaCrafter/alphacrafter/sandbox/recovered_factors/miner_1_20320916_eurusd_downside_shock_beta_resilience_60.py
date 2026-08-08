"""One idea: EURUSD-downside-shock beta resilience, 60 sessions.
Measures an asset's return beta to EURUSD only when EURUSD has an unusually
negative daily move, minus ordinary EURUSD beta.  The difference identifies
cross-asset relative resilience in dollar/risk-off currency shocks.  Signal
uses only completed information and is lagged one session."""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
d=get_index_daily_data('EURUSD',5000).copy(); d.date=pd.to_datetime(d.date)
eur=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce').reindex(P.index).ffill(); er=eur.pct_change()
def beta(x,y,w=60): return x.rolling(w,min_periods=20).cov(y)/y.rolling(w,min_periods=20).var()
# Event is formed from historical rolling distribution; current session's event
# contributes only to tomorrow's signal via final lag.
event=er < er.rolling(60,min_periods=40).quantile(.25)
def eventbeta(x,y,e,w=60):
 q=pd.concat([x.rename('x'),y.rename('y'),e.rename('e')],axis=1).where(lambda v:v.e)
 return q.x.rolling(w,min_periods=12).cov(q.y)/q.y.rolling(w,min_periods=12).var()
raw=pd.DataFrame({a:eventbeta(r[a],er,event,60)-beta(r[a],er,60) for a in A})
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
fw={k:P.shift(-k)/P-1 for k in (1,5,10,20)}
def stats(k,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1];]; y=fw[k].reindex(x.index); vals=[]; br=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);br.append(len(q))
 vals=np.asarray(vals)
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(br)),3),'min_breadth':int(min(br))} if len(vals)>1 else {'dates':len(vals)}
print('FACTOR eurusd_downside_shock_beta_resilience_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for k in (1,5,10,20): print('H',k,stats(k))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,stats(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx: mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
