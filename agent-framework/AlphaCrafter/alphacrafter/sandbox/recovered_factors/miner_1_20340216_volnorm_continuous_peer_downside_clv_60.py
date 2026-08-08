"""One idea: volatility-normalized continuous peer-downside close-location reversal (60d)."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff,cs=z['P'],z['r'],z['S'],z['A'],z['cutoff'],z['cs']
H={};L={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 H[a]=pd.to_numeric(d.high,errors='coerce');L[a]=pd.to_numeric(d.low,errors='coerce')
h=pd.DataFrame(H).reindex(P.index);l=pd.DataFrame(L).reindex(P.index)
clv=((2*P-h-l)/(h-l).replace(0,np.nan)).clip(-1,1)
peer=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A})
# Weight each close-location observation by continuous, volatility-normalized
# peer downside. Normalizing the signal by own trailing return volatility makes
# a unit of intraday resilience comparable across structurally volatile assets.
ps=peer.rolling(60,min_periods=40).std().replace(0,np.nan);w=(-peer/ps).clip(0,4).fillna(0)
base=-(clv.mul(w).rolling(60,min_periods=35).sum()/w.rolling(60,min_periods=35).sum().replace(0,np.nan)).where(w.rolling(60,min_periods=35).sum()>=8)
ov=r.rolling(20,min_periods=15).std().replace(0,np.nan)
raw=cs(base/ov)
def resid(x,controls):
 out=pd.DataFrame(np.nan,index=x.index,columns=x.columns)
 for d in x.index:
  q=pd.concat([x.loc[d].rename('y')]+[v.loc[d].rename(str(i)) for i,v in enumerate(controls)],axis=1).dropna()
  if len(q)<8:continue
  X=np.c_[np.ones(len(q)),q.iloc[:,1:].to_numpy(float)]
  if np.linalg.matrix_rank(X)<X.shape[1]:continue
  out.loc[d,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
 return out
# reconstruct the two admitted precursor signals exactly enough for the audit
extreme=peer<peer.shift(1).rolling(60,min_periods=40).quantile(.2)
event=cs(-clv.where(extreme).rolling(60,min_periods=12).mean().where(clv.where(extreme).notna().rolling(60,min_periods=12).sum()>=12))
prior=resid(event,[S['inverse_extreme_broad_weakness_magnitude_weighted_relative_capture_60']])
continuous=resid(cs(base),[S['inverse_extreme_broad_weakness_magnitude_weighted_relative_capture_60'],S['conditional_downside_participation_avoidance_60']])
S['residualized_inverse_extreme_peer_downside_close_location_reversal_60']=cs(prior).shift(1)
S['continuous_peer_downside_intensity_close_location_reversal_60']=cs(continuous).shift(1)
cand=cs(resid(raw,[S['inverse_extreme_broad_weakness_magnitude_weighted_relative_capture_60'],S['conditional_downside_participation_avoidance_60'],S['continuous_peer_downside_intensity_close_location_reversal_60']])).shift(1)
fw={k:P.shift(-k)/P-1 for k in (1,5,10,20)}
def stat(k,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1];];y=fw[k].reindex(x.index);v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(rho):v.append(rho);b.append(len(q))
 if not v:return {'dates':0}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR volnorm_continuous_peer_downside_close_location_reversal_60 CUTOFF',cutoff.date(),'ASSETS',len(A),'LIBRARY',len(S))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
for k in (1,5,10,20):print('H',k,stat(k))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stat(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 if len(q)<8:continue
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 if np.isfinite(rho) and abs(rho)>mx:mx,who,ev=abs(rho),n,len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev)
