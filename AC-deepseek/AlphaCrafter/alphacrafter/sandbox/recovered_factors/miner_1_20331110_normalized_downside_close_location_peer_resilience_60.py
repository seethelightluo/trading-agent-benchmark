"""One idea: normalized downside close-location peer resilience over 60 sessions."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
from alphacrafter.sim.utils import get_stock_daily_data
H={};L={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 H[a]=pd.to_numeric(d.high,errors='coerce'); L[a]=pd.to_numeric(d.low,errors='coerce')
H=pd.DataFrame(H).reindex(P.index);L=pd.DataFrame(L).reindex(P.index)
# A close near its daily high while peers are weak indicates downside absorption.
# Normalize each asset versus its own trailing distribution, then take its 60-day
# mean only on completed broad-negative peer sessions; lag one full session.
clv=(P-L)/(H-L).replace(0,np.nan)
clvz=(clv-clv.rolling(60,min_periods=40).mean())/clv.rolling(60,min_periods=40).std().replace(0,np.nan)
peer=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A})
cand=clvz.where(peer<0).rolling(60,min_periods=15).mean()
cand=cand.sub(cand.median(axis=1),axis=0).shift(1)
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
print('FACTOR normalized_downside_close_location_peer_resilience_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in(1,5,10,20):print('H',h,stat(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stat(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
