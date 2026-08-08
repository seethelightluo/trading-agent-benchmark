"""One idea: peer-dispersion-weighted close-location reversal (60d).
Tests whether an asset's intraday close location reverses after high cross-asset disagreement,
rather than directional peer upside/downside episodes."""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff,cs=z['P'],z['r'],z['S'],z['A'],z['cutoff'],z['cs']
H={}; L={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 H[a]=pd.to_numeric(d.high,errors='coerce'); L[a]=pd.to_numeric(d.low,errors='coerce')
h=pd.DataFrame(H).reindex(P.index); l=pd.DataFrame(L).reindex(P.index)
clv=((2*P-h-l)/(h-l).replace(0,np.nan)).clip(-1,1)
# Each asset sees dispersion among the other fourteen returns; standardize its intensity locally.
peer_disp=pd.DataFrame({a:r.drop(columns=a).std(axis=1) for a in A})
scale=peer_disp.rolling(60,min_periods=40).median().replace(0,np.nan)
w=(peer_disp/scale).clip(.25,4).fillna(0)
raw=clv.mul(w).rolling(60,min_periods=35).sum()/w.rolling(60,min_periods=35).sum().replace(0,np.nan)
raw=raw.where(w.rolling(60,min_periods=35).count()>=35)
ov=r.rolling(20,min_periods=15).std().replace(0,np.nan)
def resid(x, controls):
 out=pd.DataFrame(np.nan,index=x.index,columns=x.columns)
 for d in x.index:
  q=pd.concat([x.loc[d].rename('y')]+[v.loc[d].rename(str(i)) for i,v in enumerate(controls)],axis=1).dropna()
  if len(q)<8: continue
  X=np.c_[np.ones(len(q)),q.iloc[:,1:].to_numpy(float)]
  if np.linalg.matrix_rank(X)<X.shape[1]: continue
  out.loc[d,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
 return out
# Explicitly strip close sibling designs to make diversification audit meaningful.
controls=[v for n,v in S.items() if ('close_location' in n or 'relative_capture' in n or 'downside_participation' in n)]
cand=cs(-resid(cs(raw/ov),controls)).shift(1)
fw={k:P.shift(-k)/P-1 for k in (1,5,10,20)}
def stat(k,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]; y=fw[k].reindex(x.index); vals=[];breadth=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   u=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(u): vals.append(u);breadth.append(len(q))
 if not vals:return {'dates':0}
 vals=np.asarray(vals);return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR peer_dispersion_weighted_close_location_reversal_60 CUTOFF',cutoff.date(),'ASSETS',len(A),'LIBRARY',len(S),'CONTROLS',len(controls))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
for k in (1,5,10,20): print('H',k,stat(k))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stat(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 if len(q)<8:continue
 u=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 if np.isfinite(u) and abs(u)>mx:mx,who,ev=abs(u),n,len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev)
