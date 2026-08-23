import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-07-25')
p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date')
 p[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
# defensive-relative residual reversal: subtract contemporaneous defensive basket 60d trend,
# then reverse residual and scale by trailing 20d volatility; all inputs lagged by construction
D=p[['XAU','US10Y','CN10Y']].pct_change(60).mean(axis=1)
sig=-(p.pct_change(60).sub(D,axis=0))/(r.rolling(20).std()*np.sqrt(252))
sig=sig.rank(axis=1,pct=True)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ds=[]; cov=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c); ds.append(dt); cov.append(len(a)/15)
 x=pd.Series(vals,index=ds)
 print('H',h,'dates',len(x),'avg_n',np.mean(np.array(cov)*15),'IC',x.mean(),'ICIR',x.mean()/x.std(),'hit',(x>0).mean(),'coverage',np.mean(cov))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-07-25')]:
  z=x.loc[a:b]; print('REG',a,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std() if len(z)>1 else np.nan)
print('turnover',sig.diff().abs().mean(axis=1).mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('scripts/miner_3_20290726_defensive_relative_signal.csv',index=False)
print('artifact rows',len(out),'latest',out.date.max())
