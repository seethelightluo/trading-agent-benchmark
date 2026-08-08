"""Miner_2 single-idea test: USDJPY-downshock residual response asymmetry.
At decision date t the factor uses only bars through t-1: it measures each asset's
60-bar idiosyncratic residual return on the day *after* a large USDJPY decline,
less its residual return on other days, standardized by residual volatility.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
src=open('scripts/miner_3_20320624_revalidate_common_stress_repair_rank_migration_60obs.py').read()
src=src.replace("END=pd.Timestamp('2032-06-23')", "END=pd.Timestamp('2034-12-06')")
exec(src.split("print('REVALIDATION common_stress")[0])
# At row t, lag 1 is the latest completed asset bar and lag 2 FX identifies
# whether that bar is the response day following an FX shock.
fx=ld('USDJPY',root='../persistent/index_data').reindex(p.index).pct_change(fill_method=None)
fx_lag=fx.shift(2)
shock=fx_lag.le(fx_lag.rolling(60,min_periods=30).quantile(.20))
b=beta(r.shift(1),m.shift(1),60); res=r.shift(1)-b.mul(m.shift(1),axis=0)
W=60; f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for t in range(W+2,len(p)):
 ix=p.index[t-W:t]
 e=shock.loc[ix]
 for a in A:
  z=res.loc[ix,a]; yes=z[e].dropna(); no=z[~e].dropna(); sd=z.std(ddof=1)
  if len(yes)>=7 and len(no)>=25 and sd>0:
   f.loc[p.index[t],a]=(yes.mean()-no.mean())/sd
print('FACTOR usdjpy_downshock_residual_response_asymmetry_60obs endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6),'downshock_events',int(shock.sum()))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H);z=[];ds=[];ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds); R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
best=max(R,key=lambda h:abs(R[h][0].mean())*abs(R[h][0].mean()/R[h][0].std(ddof=1))); z,ds,_=R[best]; print('SELECTED',best)
for nm,st,en in [('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('recent_2033_34','2033-01-01','2034-12-06')]:
 q=z[(ds>=st)&(ds<=en)];print('REGIME',nm,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),6))
rnk=f.rank(axis=1,pct=True); ch=(rnk-rnk.shift()).abs().stack(); print('QUALITY rank_change_turnover',round(ch.mean(),6),'comparisons',len(ch),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
# Reconstruct all active/historical library definitions for binding novelty evidence.
audit=src[src.index('# Full current-library reconstruction'):];audit=audit[:audit.index('mx=-1')];exec(audit)
mx=-1;who=None;cells=0;missing=[]
for n,s in L.items():
 q=pd.concat([f.stack(),s.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if not len(q): missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx: mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)
f.to_pickle('scripts/miner_2_20341207_usdjpy_downshock_residual_response_asymmetry_60obs_signal.pkl')
