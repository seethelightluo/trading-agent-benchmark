"""One idea: inverse peer-relative downside overnight-gap frequency, 40 sessions."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
gap=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date').loc[lambda x:x.index<=cutoff]
 gap[a]=pd.to_numeric(d.open,errors='coerce').reindex(P.index)/pd.to_numeric(d.close,errors='coerce').reindex(P.index).shift(1)-1
peer=gap.median(axis=1)
event=(gap.lt(peer,axis=0)&gap.lt(0)).astype(float).where(gap.notna())
cand=(-event.rolling(40,min_periods=24).mean()).sub((-event.rolling(40,min_periods=24).mean()).median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]; y=fw[h].reindex(x.index); vals=[];bread=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);bread.append(len(q))
 v=np.array(vals)
 return {'dates':len(v),'ic':round(v.mean(),6),'icir':round(v.mean()/v.std(ddof=1),6),'hit':round((v>0).mean(),6),'mean_breadth':round(np.mean(bread),3),'min_breadth':min(bread)} if len(v)>1 else {'dates':len(v)}
print('FACTOR inverse_relative_downside_overnight_gap_frequency_40 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',cand.notna().sum().sum(),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(cand.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(cand.std(axis=1).mean(),6))
for h in (1,5,10,20): print('H',h,stat(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,stat(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(rho,6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
