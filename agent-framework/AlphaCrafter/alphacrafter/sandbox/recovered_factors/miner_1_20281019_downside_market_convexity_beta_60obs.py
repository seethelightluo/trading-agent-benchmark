"""One idea: downside-market convexity beta, a 60-observation exposure to the magnitude (not direction) of broad cross-asset selloffs."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
# Reuse the exact data loading and active-library signal reconstructions, but not its prior candidate/report.
src=open('scripts/miner_1_20280810_intraday_close_location_persistence_20obs.py').read()
src=src.replace("END='2028-08-09'", "END='2028-10-18'")
exec(src.split('eval_dates=')[0])
# One candidate: rolling slope of each asset return on the squared magnitude of negative equal-weight/median market returns.
# Positive values identify assets that historically retain/improve return as common selloffs become more severe.
down_mag=(-med.clip(upper=0)).pow(2)
w=60; n=down_mag.notna().astype(float).rolling(w,min_periods=45).sum(); sx=down_mag.rolling(w,min_periods=45).sum(); den=down_mag.pow(2).rolling(w,min_periods=45).sum()-sx*sx/n
f=pd.DataFrame({a:((down_mag*r[a]).rolling(w,min_periods=45).sum()-sx*r[a].rolling(w,min_periods=45).sum()/n)/den for a in A})
f=f.replace([np.inf,-np.inf],np.nan)
eval_dates=f.index[f.notna().sum(axis=1)>=8]; allh={}
print('FACTOR downside_market_convexity_beta_60obs visible_through',END,'assets',len(A),'signal_cells',int(f.notna().sum().sum()),'/',f.size)
for h in [1,5,10,20]:
 vals=[]; ns=[]; y=c.shift(-h)/c-1
 for t in eval_dates:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8: vals.append((t,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic));ns.append(len(q))
 s=pd.Series(dict(vals));allh[h]=s
 print('H',h,'dates',len(s),'IC %.6f ICIR %.6f hit %.4f mean_instruments %.2f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),np.mean(ns)))
bh=max(allh,key=lambda h:abs(allh[h].mean())*abs(allh[h].mean()/allh[h].std(ddof=1)));s=allh[bh];print('SELECTED',bh)
for x,y,nm in [('2020','2022','2020-21'),('2022','2024','2022-23'),('2024','2026','2024-25'),('2026','2030','2026-current')]:
 q=s[(s.index>=x)&(s.index<y)];print('REGIME',nm,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
rk=f.rank(axis=1);to=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i],rk.iloc[i-1]],axis=1).dropna()
 if len(q)>=8: to.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('TURNOVER',round(np.mean(to),6),'coverage',round(f.notna().mean().mean(),4))
mx=(-1,None,0);evidence=0
for name,g in L.items():
 vals=[]
 for t in eval_dates:
  q=pd.concat([f.loc[t],g.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8: vals.append(abs(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
 k=max(vals) if vals else np.nan; print('LIB',name,'max_abs_rho',k,'dates',len(vals)); evidence+=len(vals)
 if np.isfinite(k) and k>mx[0]:mx=(k,name,len(vals))
print('MAX_ABS_LIBRARY_CORRELATION %.6f closest %s dates %d evidence_cells %d'%(*mx,evidence))
