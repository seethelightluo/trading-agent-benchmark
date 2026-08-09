import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); r2=pd.DataFrame({a:p[a].pct_change(2) for a in A})
resid=r2.sub(r2.median(axis=1),axis=0); vol=r.rolling(20,min_periods=10).std(); raw=-resid.div(vol,axis=0)
rows=[]; sig=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 vals={a:raw.at[dt,a] if dt in raw.index else np.nan for a in A}; good=[v for v in vals.values() if np.isfinite(v)]; med=np.nanmedian(good) if len(good)>=8 else np.nan
 for a in A: sig.append((dt,a,vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
 f=[];y=[]
 for a in A:
  if dt not in p[a].index: continue
  i=p[a].index.get_loc(dt); z=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
  if np.isfinite(z) and i+1<len(p[a]): f.append(z);y.append(p[a].iloc[i+1]/p[a].iloc[i]-1)
 if len(f)>=8: rows.append((dt,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','ic','n']); print('dates',len(d),'avg_n',round(d.n.mean(),2),'IC',round(d.ic.mean(),6),'ICIR',round(d.ic.mean()/d.ic.std(),6),'hit',round((d.ic>0).mean(),4))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 q=d.set_index('date').loc[lo:hi].ic;print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else None,round((q>0).mean(),4))
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_2_20270225_volscaled_residual_reversal2.csv',index=False);print('artifact',len(out)); print('turnover',round(out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
