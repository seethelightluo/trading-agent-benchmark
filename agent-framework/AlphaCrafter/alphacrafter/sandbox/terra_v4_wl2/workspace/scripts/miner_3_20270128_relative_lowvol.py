import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r={a:p[a].pct_change() for a in A}; vol={a:r[a].rolling(20,min_periods=15).std() for a in A}
idx=sorted(set().union(*[set(x.index) for x in p.values()])); rows=[]; sig=[]
for d in idx:
 vals={a:vol[a].get(d,np.nan) for a in A}; med=np.nanmedian([v for v in vals.values() if np.isfinite(v)]) if sum(np.isfinite(list(vals.values())))>=8 else np.nan
 for a in A:
  f=-(vals[a]/med) if np.isfinite(vals[a]) and np.isfinite(med) else np.nan; sig.append((d,a,f))
 for h in [1,5,10]:
  x=[];y=[]
  for a in A:
   if d not in p[a].index or not np.isfinite(vals[a]) or not np.isfinite(med):continue
   i=p[a].index.get_loc(d)
   if i+h>=len(p[a]):continue
   yy=p[a].iloc[i+h]/p[a].iloc[i]-1
   if np.isfinite(yy):x.append(-(vals[a]/med));y.append(yy)
  if len(x)>=8:rows.append((d,h,spearmanr(x,y).statistic,len(x)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=df[df.h==h];print('H',h,'dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(),'hit',(x.ic>0).mean())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic;print(lo,len(z),round(z.mean(),5),round(z.mean()/z.std(),5))
pd.DataFrame(sig,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_3_20270128_relative_lowvol.csv',index=False)
wide=pd.DataFrame(sig,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal');print('turnover',wide.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
