"""One idea: dispersion-conditioned downside close-location resilience (60 sessions)."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
# Loads only visible daily history, full active-library reconstruction.
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff=z['P'],z['S'],z['A'],z['cutoff']
r=P.pct_change(); med=r.median(axis=1); disp=r.std(axis=1)
H=pd.DataFrame(index=P.index,columns=A,dtype=float); L=H.copy(); C=H.copy()
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 H[a]=pd.to_numeric(d.high,errors='coerce').reindex(P.index);L[a]=pd.to_numeric(d.low,errors='coerce').reindex(P.index);C[a]=pd.to_numeric(d.close,errors='coerce').reindex(P.index)
# On sessions with elevated cross-asset dispersion and an asset-specific loss,
# close location measures whether that asset resisted selling into the close.
loc=(C-L)/(H-L).replace(0,np.nan)
highdisp=disp>disp.rolling(60,min_periods=40).quantile(.70)
event=pd.DataFrame({a:highdisp & (r[a]<med) for a in A})
raw=loc.where(event).rolling(60,min_periods=15).mean()
# Remove contemporaneous 20d risk-adjusted trend cross-sectionally, then lag one completed session.
trend=P.pct_change(20)/r.rolling(20,min_periods=15).std()
def resid(x,b):
 out=pd.DataFrame(np.nan,index=x.index,columns=x.columns)
 for dt in x.index:
  q=pd.concat([x.loc[dt],b.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,1].var()>1e-12:
   be=np.cov(q.iloc[:,0],q.iloc[:,1],ddof=1)[0,1]/q.iloc[:,1].var()
   out.loc[dt,q.index]=q.iloc[:,0]-q.iloc[:,0].mean()-be*(q.iloc[:,1]-q.iloc[:,1].mean())
 return out
cand=resid(raw,trend).shift(1)
fw={h:P.pct_change(h).shift(-h) for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];y=fw[h].reindex(x.index);v=[];n=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(rho):v.append(rho);n.append(len(q))
 if len(v)<2:return {'dates':len(v)}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'breadth':round(float(np.mean(n)),3),'min_breadth':min(n)}
print('FACTOR dispersion_conditioned_downside_close_location_resilience_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for name,period in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:
 print('REGIME20',name,stats(20,period))
mx=-1;who='';ev=0
for name,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 if len(q)>=8:
  rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(rho) and abs(rho)>mx:mx=abs(float(rho));who=name;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
