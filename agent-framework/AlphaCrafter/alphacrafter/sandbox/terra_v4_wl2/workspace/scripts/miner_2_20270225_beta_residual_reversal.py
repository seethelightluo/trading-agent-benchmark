import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); m=r.median(axis=1)
# Orthogonal short reversal: 3d relative return after removing rolling 60d asset beta to common cross-asset return.
rel=r.sub(m,axis=0); cov=rel.rolling(60,min_periods=30).cov(m); vm=m.rolling(60,min_periods=30).var()
beta=cov.div(vm,axis=0); resid3=rel.rolling(3,min_periods=3).sum()-beta.mul(m.rolling(3,min_periods=3).sum(),axis=0)
vol=rel.rolling(20,min_periods=10).std(); raw=-resid3.div(vol,axis=0)
rows=[]; sig=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 vals=raw.loc[dt] if dt in raw.index else pd.Series(dtype=float); med=vals.dropna().median(); good=vals.dropna()
 for a in A:sig.append((dt,a,vals.get(a,np.nan)-med if len(good)>=8 and np.isfinite(vals.get(a,np.nan)) else np.nan))
 f=[];y=[]
 for a in A:
  if dt not in p[a].index: continue
  z=vals.get(a,np.nan)-med
  i=p[a].index.get_loc(dt)
  if len(good)>=8 and np.isfinite(z) and i+1<len(p[a]): f.append(z); y.append(p[a].iloc[i+1]/p[a].iloc[i]-1)
 if len(f)>=8: rows.append((dt,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','ic','n']); print('dates',len(d),'avg_n',d.n.mean(),'IC',d.ic.mean(),'ICIR',d.ic.mean()/d.ic.std(),'hit',(d.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 q=d.set_index('date').loc[lo:hi].ic; print(lo,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
out=pd.DataFrame(sig,columns=['date','asset','signal']); out.to_csv('../persistent/factor_signals_miner_2_20270225_beta_residual_reversal.csv',index=False); print('coverage',out.signal.notna().mean(),'turn',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
