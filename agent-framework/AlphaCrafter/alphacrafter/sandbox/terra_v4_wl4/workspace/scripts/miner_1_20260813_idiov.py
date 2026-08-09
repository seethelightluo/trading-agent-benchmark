import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); D[s]=d.set_index('date').close.pct_change()
idx=sorted(set.intersection(*[set(x.index) for x in D.values()])); R=pd.DataFrame({s:D[s] for s in U},index=idx); m=R.mean(axis=1)
for h in [20,60]:
 out=[]
 for i in range(h,len(R)-1):
  q=R.iloc[i-h:i]; mm=m.iloc[i-h:i]; vals={}; fw={}
  for s in U:
   y=q[s]; ok=y.notna()&mm.notna()
   if ok.sum()<max(10,h//2): continue
   beta=np.cov(y[ok],mm[ok],ddof=1)[0,1]/np.var(mm[ok],ddof=1) if np.var(mm[ok])>1e-12 else np.nan
   resid=y[ok]-beta*mm[ok]; vals[s]=-resid.std(); fw[s]=R.iloc[i+1][s]
  if len(vals)>=8: out.append(spearmanr(pd.Series(vals),pd.Series(fw).reindex(vals)).statistic)
 z=pd.Series(out).dropna();print(h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean())
