"""Repair audit: inverse residual intraday demand persistence (10/60)."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
# Latest maintained reconstruction includes the active-library signal definitions.
src=open('scripts/miner_2_20340427_normal_dispersion_residual_reversal_persistence_60obs.py').read()
src=src.replace("END=pd.Timestamp('2032-06-23')", "END=pd.Timestamp('2034-04-26')")
exec(src.split("# In quiet")[0])
# Explicitly load open prices; the earlier failed audit only reconstructed close panels.
op=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'open'].replace(0,np.nan).reindex(p.index) for a in A})
intra=p.div(op)-1
common=intra.median(axis=1)
beta=intra.rolling(60,min_periods=45).cov(common).div(common.rolling(60,min_periods=45).var()+1e-12,axis=0)
f=-(intra-beta.mul(common,axis=0)).rolling(10,min_periods=8).mean()
print('CANDIDATE inverse_residual_intraday_demand_persistence_10_60obs endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H); z=[]; ds=[]; ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.asarray(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'mean_names',round(np.mean(ns),3),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
for H,(z,ds,_) in R.items():
 for nm,st,en in [('2020_2027','2020-01-01','2027-12-31'),('2028_2030','2028-01-01','2030-12-31'),('2031_now','2031-01-01','2034-04-26'),('latest_6m','2033-10-26','2034-04-26')]:
  x=z[(ds>=st)&(ds<=en)];print('REGIME H',H,nm,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
r=f.rank(axis=1,pct=True); print('QUALITY rank_turnover',round(r.diff().abs().stack().mean(),6),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
# Rebuild the maintained library and append definitions added after the inherited source.
audit=src[src.index('# Full current-library reconstruction'):];exec(audit[:audit.index('mx=-1')])
hi_state=disp.shift(1).gt(disp.rolling(60,min_periods=45).median().shift(1)).astype(float);den=hi_state.rolling(60,min_periods=45).sum()
L['dispersion_conditioned_residual_resilience_60obs']=res.shift(1).rank(axis=1,pct=True).mul(hi_state,axis=0).rolling(60,min_periods=45).sum().div(den.replace(0,np.nan),axis=0).where(den>=12)
resolution=(disp.shift(1)>disp.rolling(60,min_periods=45).median().shift(1))&(disp<disp.rolling(20,min_periods=15).median())
L['dispersion_resolution_residual_rebound_60obs']=res.shift(1).rank(axis=1,pct=True).where(resolution).rolling(60,min_periods=45).mean()
mx=-1;who=None;cells=0;missing=[]
for n,s in L.items():
 q=pd.concat([f.stack(),s.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)<8:missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if not np.isfinite(rho):missing.append(n);continue
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'closest',who,'evidence_cells',cells,'signals_tested',len(L),'missing',missing)
