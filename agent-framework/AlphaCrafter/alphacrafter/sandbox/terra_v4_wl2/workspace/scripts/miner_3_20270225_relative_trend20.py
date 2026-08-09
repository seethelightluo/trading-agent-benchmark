import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A})
# Orthogonal medium trend: asset 20-day return relative to contemporaneous cross-asset median.
ret20=pd.DataFrame({a:p[a].pct_change(20) for a in A}); peer=ret20.median(axis=1); raw=ret20.sub(peer,axis=0)
rows=[]; sig=[]
for dt in ret20.index:
 vals=raw.loc[dt]; good=vals.dropna();
 if len(good)<8: continue
 # rank-preserving signal; point-in-time and forward one day
 f=[]; y=[]
 for a in A:
  if a not in vals or not np.isfinite(vals[a]) or dt not in p[a].index: continue
  i=p[a].index.get_loc(dt)
  if i+1<len(p[a]): f.append(vals[a]); y.append(p[a].iloc[i+1]/p[a].iloc[i]-1); sig.append((dt,a,vals[a]))
 if len(f)>=8: rows.append((dt,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','ic','n']); print('factor=relative_trend20');print('dates',len(d),'avg_n',d.n.mean(),'IC',d.ic.mean(),'ICIR',d.ic.mean()/d.ic.std(),'hit',(d.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 q=d.set_index('date').loc[lo:hi].ic; print(lo,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_relative_trend20.csv',index=False)
print('coverage',len(out)/((ret20.index.max()-ret20.index.min()).days/365*252*15),'turn',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
