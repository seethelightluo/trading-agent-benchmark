"""One idea: residualized inverse extreme-peer-downside close-location reversal, 60 sessions.
Residualizes the previous CLV reversal cross-sectionally against its empirically overlapping
extreme broad-weakness relative-capture library factor. Both are lagged one completed session."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
other=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A}); f={}
for a in A:
 d=z['get_stock_daily_data'](a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date').loc[lambda x:x.index<=cutoff]
 c=pd.to_numeric(d.close,errors='coerce');h=pd.to_numeric(d.high,errors='coerce');l=pd.to_numeric(d.low,errors='coerce');clv=((2*c-h-l)/(h-l).replace(0,np.nan)).reindex(P.index)
 th=other[a].shift(1).rolling(60,min_periods=40).quantile(.20);f[a]=clv.where(other[a]<th).rolling(60,min_periods=12).mean()
base=-pd.DataFrame(f).sub(pd.DataFrame(f).median(axis=1),axis=0).shift(1)
control=S['inverse_extreme_broad_weakness_magnitude_weighted_relative_capture_60'];cand=pd.DataFrame(index=P.index,columns=A,dtype=float)
for dt in P.index:
 q=pd.concat([base.loc[dt].rename('y'),control.loc[dt].rename('x')],axis=1).dropna()
 if len(q)>=8:
  x=q.x.to_numpy(float); y=q.y.to_numpy(float); b=np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean()) if np.dot(x-x.mean(),x-x.mean())>0 else 0
  cand.loc[dt,q.index]=y-(y.mean()+b*(x-x.mean()))
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];y=fw[h].reindex(x.index);v=[];b=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   u=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(u):v.append(u);b.append(len(q))
 v=np.asarray(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))} if len(v)>1 else {'dates':len(v)}
print('FACTOR residualized_inverse_extreme_peer_downside_close_location_reversal_60 CUTOFF',cutoff.date(),'ASSETS',len(A),'CONTROL inverse_extreme_broad_weakness_magnitude_weighted_relative_capture_60')
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
