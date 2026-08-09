"""One idea: trend-orthogonal peer-relative overnight/intraday resilience asymmetry."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff=z['P'],z['S'],z['A'],z['cutoff']
O=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 O[a]=pd.to_numeric(d.sort_values('date').set_index('date').open,errors='coerce').reindex(P.index)
r=P.pct_change(); og=O/P.shift()-1; ir=P/O-1
raw=(og.sub(og.median(axis=1),axis=0)-ir.sub(ir.median(axis=1),axis=0)).rolling(40,min_periods=24).mean()
# Per-date OLS residual after projecting the continuous asymmetry on 20-day risk-adjusted trend.
trend=P.pct_change(20)/r.rolling(20,min_periods=15).std()
def residual(x,b):
 out=pd.DataFrame(np.nan,index=x.index,columns=x.columns)
 for d in x.index:
  q=pd.concat([x.loc[d],b.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,1].var()>1e-14:
   slope=np.cov(q.iloc[:,0],q.iloc[:,1],ddof=1)[0,1]/q.iloc[:,1].var()
   out.loc[q.index,d] if False else None
   out.loc[d,q.index]=q.iloc[:,0]-q.iloc[:,0].mean()-slope*(q.iloc[:,1]-q.iloc[:,1].mean())
 return out
cand=residual(raw,trend).shift(1)
fw={h:P.pct_change(h).shift(-h) for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]; y=fw[h].reindex(x.index); vals=[]; br=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):vals.append(v);br.append(len(q))
 if len(vals)<2:return {'dates':len(vals)}
 v=np.array(vals);return {'dates':len(v),'ic':round(v.mean(),6),'icir':round(v.mean()/v.std(ddof=1),6),'hit':round((v>0).mean(),6),'breadth':round(np.mean(br),3),'min_breadth':min(br)}
print('FACTOR trend_orthogonal_overnight_intraday_resilience_asymmetry_40 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',cand.notna().sum().sum(),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(cand.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(cand.std(axis=1).mean(),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
