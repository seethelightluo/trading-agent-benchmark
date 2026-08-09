import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); med=r.median(axis=1); disp=r.sub(med,axis=0).abs().median(axis=1)
# high-dispersion residual reversal, 20d percentile estimated using history through t
thr=disp.rolling(60,min_periods=30).quantile(.70)
sig=[]; rows=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 if dt not in thr.index or not np.isfinite(thr.get(dt,np.nan)) or disp.get(dt,0)<=thr[dt]: continue
 vals={a: (r.at[dt,a]-med[dt]) if dt in r.index and np.isfinite(r.at[dt,a]) else np.nan for a in A}
 good=[v for v in vals.values() if np.isfinite(v)]
 if len(good)<8: continue
 # residual reversal normalized by 20d idio vol
 f=[];y=[]
 for a in A:
  if dt not in p[a].index: continue
  i=p[a].index.get_loc(dt); vol=r[a].sub(med,axis=0).rolling(20,min_periods=10).std().get(dt,np.nan)
  z=-vals[a]/vol if np.isfinite(vals[a]) and np.isfinite(vol) and vol>0 else np.nan
  sig.append((dt,a,z))
  if np.isfinite(z) and i+1<len(p[a]): f.append(z);y.append(p[a].iloc[i+1]/p[a].iloc[i]-1)
 if len(f)>=8: rows.append((dt,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','ic','n']);print('active_dates',len(d),'avg_n',d.n.mean(),'IC',d.ic.mean(),'ICIR',d.ic.mean()/d.ic.std(),'hit',(d.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 q=d.set_index('date').loc[lo:hi].ic; print(lo,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_2_20270225_highdisp_residrev.csv',index=False)
print('coverage',len(d)/len(set().union(*[set(x.index) for x in p.values()])),'turnover',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
