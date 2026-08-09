"""One idea: continuous peer-downside-intensity weighted close-location reversal, residualized against its closest event-capture antecedents."""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff,cs,rel=z['P'],z['r'],z['S'],z['A'],z['cutoff'],z['cs'],z['rel']
# Close location measures where within its daily range an asset finished.  Weight only
# broad peer-downside intensity continuously, avoiding sparse hard event selection.
from alphacrafter.sim.utils import get_stock_daily_data
H={};L={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 H[a]=pd.to_numeric(d.high,errors='coerce'); L[a]=pd.to_numeric(d.low,errors='coerce')
h,l=pd.DataFrame(H).reindex(P.index),pd.DataFrame(L).reindex(P.index)
clv=((2*P-h-l)/(h-l).replace(0,np.nan)).clip(-1,1)
peer=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A})
scale=peer.rolling(60,min_periods=40).std().replace(0,np.nan)
w=(-peer/scale).clip(lower=0,upper=4).fillna(0)
# weighted average needs sufficient nonzero stress mass in trailing 60 sessions
num=clv.mul(w).rolling(60,min_periods=35).sum(); den=w.rolling(60,min_periods=35).sum().replace(0,np.nan)
raw=cs(-(num/den).where(den>=8))
# Remove two mechanistically nearest existing downside-capture factors cross-sectionally.
def residual(x, controls):
 out=pd.DataFrame(np.nan,index=x.index,columns=x.columns)
 for dt in x.index:
  q=pd.concat([x.loc[dt].rename('y')]+[c.loc[dt].rename(str(i)) for i,c in enumerate(controls)],axis=1).dropna()
  if len(q)<8: continue
  X=np.column_stack([np.ones(len(q)),q.iloc[:,1:].to_numpy(float)])
  if np.linalg.matrix_rank(X)<X.shape[1]: continue
  b=np.linalg.lstsq(X,q.y.to_numpy(float),rcond=None)[0]
  out.loc[dt,q.index]=q.y.to_numpy(float)-X@b
 return out
cand=cs(residual(raw,[S['inverse_extreme_broad_weakness_magnitude_weighted_relative_capture_60'],S['conditional_downside_participation_avoidance_60']])).shift(1)
# Include the Jan-2034 admitted residual CLV factor in the binding correlation audit.
extreme=peer < peer.shift(1).rolling(60,min_periods=40).quantile(.2)
event_clv=clv.where(extreme); prior=cs(-event_clv.rolling(60,min_periods=12).mean().where(event_clv.notna().rolling(60,min_periods=12).sum()>=12))
S['residualized_inverse_extreme_peer_downside_close_location_reversal_60']=cs(residual(prior,[S['inverse_extreme_broad_weakness_magnitude_weighted_relative_capture_60']])).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]; y=fw[h].reindex(x.index); vals=[]; breadth=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v); breadth.append(len(q))
 if not vals:return {'dates':0}
 vals=np.array(vals); return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR continuous_peer_downside_intensity_close_location_reversal_60 CUTOFF',cutoff.date(),'ASSETS',len(A),'LIBRARY',len(S))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20): print('H',h,stat(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stat(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx,who,ev=abs(rho),n,len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev)
