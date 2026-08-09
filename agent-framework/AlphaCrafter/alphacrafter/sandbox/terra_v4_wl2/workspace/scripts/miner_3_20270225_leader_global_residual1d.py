import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; L=['SPX','N225','SX5E','SOX','NDX','000300.SH','000688.SH']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A};r=pd.DataFrame({a:p[a].pct_change() for a in A}); lm=r[L].median(axis=1); gm=r.median(axis=1); signal=lm-gm
rows=[]; ss=[]
for dt in signal.index:
 f=[];y=[]
 for a in A:
  if dt not in p[a].index or not np.isfinite(signal.loc[dt]):continue
  i=p[a].index.get_loc(dt)
  if i+1<len(p[a]):
   # target-specific leader residual: leave target out of global median
   others=[x for x in A if x!=a and np.isfinite(r.at[dt,x])]
   g=np.median([r.at[dt,x] for x in others]) if len(others)>=8 else gm.loc[dt]
   z=lm.loc[dt]-g; f.append(z); y.append(p[a].iloc[i+1]/p[a].iloc[i]-1); ss.append((dt,a,z))
 if len(f)>=8:
  z=spearmanr(f,y).statistic
  if np.isfinite(z):rows.append((dt,z,len(f)))
d=pd.DataFrame(rows,columns=['date','ic','n']);print('factor=leader-minus-global-shock1d');print('dates',len(d),'avg_n',d.n.mean(),'IC',d.ic.mean(),'ICIR',d.ic.mean()/d.ic.std(),'hit',(d.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 q=d.set_index('date').loc[lo:hi].ic;print(lo,len(q),q.mean(),q.mean()/q.std())
out=pd.DataFrame(ss,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_leader_global_residual1d.csv',index=False);print('turn',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
