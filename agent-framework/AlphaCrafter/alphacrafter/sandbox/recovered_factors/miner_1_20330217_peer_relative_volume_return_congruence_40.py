"""One idea: peer-relative volume-return congruence 40 sessions.
Assets whose idiosyncratic returns have consistently occurred with above-peer
volume may reflect persistent cross-asset information participation."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff,V=z['P'],z['S'],z['A'],z['cutoff'],z['V']
r=P.pct_change(); rel=r.sub(r.median(axis=1),axis=0)
# Relative log-volume avoids scale differences across the deliberately heterogeneous assets.
lv=np.log(V).replace([np.inf,-np.inf],np.nan); vrel=lv.sub(lv.median(axis=1),axis=0)
raw=pd.DataFrame({a:rel[a].rolling(40,min_periods=25).corr(vrel[a]) for a in A})
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); vals=[]; br=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   qv=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(qv): vals.append(qv);br.append(len(q))
 if len(vals)<2:return {'dates':len(vals)}
 vals=np.array(vals);return {'dates':len(vals),'ic':round(vals.mean(),6),'icir':round(vals.mean()/vals.std(ddof=1),6),'hit':round((vals>0).mean(),6),'breadth':round(np.mean(br),3),'min_breadth':min(br)}
print('FACTOR peer_relative_volume_return_congruence_40 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',cand.notna().sum().sum(),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(cand.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(cand.std(axis=1).mean(),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
