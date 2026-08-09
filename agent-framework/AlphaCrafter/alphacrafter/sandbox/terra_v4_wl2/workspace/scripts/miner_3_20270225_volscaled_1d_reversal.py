import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); vol=r.rolling(20,min_periods=10).std()
# standardized one-day reversal, cross-sectional demeaned
raw=-r.div(vol,axis=0); rows=[]; sig=[]
for dt in r.index:
 vals=raw.loc[dt]; good=vals.dropna(); med=good.median() if len(good)>=8 else np.nan
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if a not in vals or not np.isfinite(vals[a]) or not np.isfinite(med): continue
   ix=p[a].index.get_loc(dt) if dt in p[a].index else -1
   if ix>=0 and ix+h<len(p[a]): f.append(vals[a]-med); y.append(p[a].iloc[ix+h]/p[a].iloc[ix]-1)
  if len(f)>=8: rows.append((dt,h,spearmanr(f,y).statistic,len(f)))
 for a in A: sig.append((dt,a,vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h];print('H',h,'dates',len(z),'avg_n',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'hit',(z.ic>0).mean())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic;print(lo,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_volscaled_1d_reversal.csv',index=False)
print('artifact',len(out))
print('turnover',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
