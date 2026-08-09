"""One idea: VIX downside-shock beta resilience (60d).
Hypothesis: assets with favorable incremental sensitivity on unusually sharp VIX
relief days can retain cross-asset relative performance after stress unwinds.
Signal is negative incremental beta (VIX-downshock beta minus all-day beta),
computed only from completed closes and lagged one session.
"""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff=z['P'],z['S'],z['A'],z['cutoff']
r=P.pct_change(); vix=z['vix']; vr=vix.pct_change()
def cs(x): return x.sub(x.median(axis=1),axis=0)
def eventbeta(x,y,event,w=60):
    q=pd.concat([x.rename('x'),y.rename('y'),event.rename('e')],axis=1).where(lambda d:d.e)
    return q.x.rolling(w,min_periods=12).cov(q.y)/q.y.rolling(w,min_periods=12).var()
# Extreme VIX declines, threshold known as of each date; compare to ordinary beta.
th=vr.rolling(60,min_periods=40).quantile(.25)
event=vr<th
allbeta=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(vr)/vr.rolling(60,min_periods=40).var() for a in A})
ebeta=pd.DataFrame({a:eventbeta(r[a],vr,event,60) for a in A})
cand=cs(-(ebeta-allbeta)).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1];]; y=fw[h].reindex(x.index); vals=[];br=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);br.append(len(q))
 vals=np.array(vals)
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(br)),3),'min_breadth':int(min(br))}
print('FACTOR inverse_vix_downside_shock_beta_resilience_60 VALIDATION_DATE 2032-12-23 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6),'EVENT_RATE',round(float(event.mean()),6))
for h in (1,5,10,20): print('H',h,stats(h))
for n,p in [('2020_22',('2020-01-01','2022-12-31')),('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME20',n,stats(20,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx: mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
