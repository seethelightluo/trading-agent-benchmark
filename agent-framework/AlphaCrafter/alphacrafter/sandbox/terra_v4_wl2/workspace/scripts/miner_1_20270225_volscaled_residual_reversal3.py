import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); common=r.median(axis=1); resid=r.sub(common,axis=0); vol=r.rolling(20,min_periods=10).std(); raw=-resid.div(vol,axis=0)
for h in [3,5,10]:
 rows=[]
 for dt in raw.index:
  vals=raw.loc[dt]; med=vals.median(skipna=True); f=[]; y=[]
  for a in A:
   if a not in p or dt not in p[a].index: continue
   i=p[a].index.get_loc(dt); z=vals[a]-med
   if np.isfinite(z) and i+h<len(p[a]): f.append(z); y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8: rows.append((dt,spearmanr(f,y).statistic,len(f)))
 d=pd.DataFrame(rows,columns=['date','ic','n']); s=d.ic
 print('h',h,'dates',len(d),'avg_n',d.n.mean(),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=d.set_index('date').loc[lo:hi].ic; print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),4))
 if h==3:
  out=pd.DataFrame({'signal':raw.sub(raw.median(axis=1),axis=0).stack()});out.to_csv('../persistent/factor_signals_miner_1_20270225_volscaled_residual_reversal3.csv')
  print('coverage',raw.notna().mean().mean(),'turn',raw.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
