import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-07-11')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date')
 d=d[d.date<=cut].set_index('date')
 px[s]=d.close
p=pd.DataFrame(px).sort_index()
r=p.pct_change()
# inverse of risk-adjusted medium trend, causal
sig=-(p.pct_change(120)/(r.rolling(20).std()*np.sqrt(252)))
# cross sectional ranks are robust and intended signal artifact
sig=sig.rank(axis=1,pct=True)
rows=[]
for h in [5,10,20]:
 f=sig; fr=p.shift(-h)/p-1
 ics=[]; dates=[]; cov=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   ics.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(dt); cov.append(len(a)/15)
 x=pd.Series(ics,index=dates).dropna()
 print('H',h,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(),'hit', (x>0).mean(),'coverage',np.mean(cov))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-07-11')]:
  z=x.loc[a:b]; print('REG',a,len(z),z.mean(),z.mean()/z.std() if len(z)>1 else np.nan)
 # turnover mean rank absolute change
 print('turnover',sig.diff().abs().mean(axis=1).mean())
# save latest signal artifact, full signal for gate provenance
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('scripts/miner_3_20290712_inverse_risk_trend_signal.csv',index=False)
print('artifact rows',len(out),'latest',out.date.max())
