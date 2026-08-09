"""One idea: continuous dispersion-weighted relative close-location resilience (60 sessions)."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
# Uses the current full admitted-library reconstruction for the mandatory correlation audit.
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py');P,S,A,cutoff=z['P'],z['S'],z['A'],z['cutoff']
r=P.pct_change(); disp=r.std(axis=1); medr=r.median(axis=1)
H={};L={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 H[a]=pd.to_numeric(d.high,errors='coerce'); L[a]=pd.to_numeric(d.low,errors='coerce')
H=pd.DataFrame(H).reindex(P.index);L=pd.DataFrame(L).reindex(P.index)
# A continuous version of the prior sparse idea: daily cross-sectional relative close
# location, weighted by normalized dispersion and by continuous own downside magnitude.
loc=(P-L)/(H-L).replace(0,np.nan)
rel_loc=loc.sub(loc.median(axis=1),axis=0)
dz=(disp/disp.rolling(60,min_periods=40).median()).clip(.25,3)
down=((medr-r).clip(lower=0)).div(disp.replace(0,np.nan),axis=0).clip(0,3)
# Numerator/denominator formulation prevents low-event coverage failure; lag prevents lookahead.
cand=(rel_loc*down.mul(dz,axis=0)).rolling(60,min_periods=35).sum()/down.mul(dz,axis=0).rolling(60,min_periods=35).sum().replace(0,np.nan)
cand=cand.shift(1)
fw={h:P.pct_change(h).shift(-h) for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];y=fw[h].reindex(x.index);v=[];n=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   u=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(u):v.append(u);n.append(len(q))
 if len(v)<2:return {'dates':len(v)}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'breadth':round(float(np.mean(n)),3),'min_breadth':min(n)}
print('FACTOR continuous_dispersion_weighted_relative_close_location_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for name,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME20',name,st(20,p))
mx=-1;who='';ev=0
for name,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 if len(q)>=8:
  u=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(u) and abs(u)>mx:mx=abs(float(u));who=name;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
