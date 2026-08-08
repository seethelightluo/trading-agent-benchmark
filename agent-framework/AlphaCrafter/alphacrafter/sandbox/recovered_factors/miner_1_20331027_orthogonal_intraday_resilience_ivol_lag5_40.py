"""One idea: idiosyncratic-volatility and lag-5-serial-dependence orthogonal intraday resilience."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff,cs=z['P'],z['r'],z['S'],z['A'],z['cutoff'],z['cs']
# Each day's intraday return (close/open), measured relative to cross-asset median.
# Average resilience is residualized cross-sectionally against inverse idiosyncratic
# volatility and inverse lag-5 peer-relative serial dependence, then lagged one day.
O={}
from alphacrafter.sim.utils import get_stock_daily_data
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);O[a]=pd.to_numeric(d.sort_values('date').set_index('date').open,errors='coerce')
O=pd.DataFrame(O).reindex(P.index)
intra=P/O-1; rel=intra.sub(intra.median(axis=1),axis=0)
base=rel.rolling(40,min_periods=25).mean()
daily=P.pct_change(); drel=daily.sub(daily.median(axis=1),axis=0)
ivol=-drel.rolling(20,min_periods=15).std()
lag5=-pd.DataFrame({a:drel[a].rolling(40,min_periods=24).corr(drel[a].shift(5)) for a in A})
def orthrow(row,x1,x2):
 q=pd.concat([row,x1,x2],axis=1).dropna()
 out=pd.Series(np.nan,index=row.index)
 if len(q)>=8 and q.iloc[:,1:].std().min()>1e-12:
  X=np.column_stack([np.ones(len(q)),q.iloc[:,1],q.iloc[:,2]])
  out.loc[q.index]=q.iloc[:,0]-X@np.linalg.lstsq(X,q.iloc[:,0],rcond=None)[0]
 return out
cand=pd.DataFrame([orthrow(base.loc[d],ivol.loc[d],lag5.loc[d]) for d in P.index],index=P.index).shift(1)
fw={h:P.shift(-h)/P-1 for h in(1,5,10,20)}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index);v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   k=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(k):v.append(k);b.append(len(q))
 if not v:return {'dates':0}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR orthogonal_intraday_resilience_ivol_lag5_40 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in(1,5,10,20):print('H',h,stat(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stat(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
