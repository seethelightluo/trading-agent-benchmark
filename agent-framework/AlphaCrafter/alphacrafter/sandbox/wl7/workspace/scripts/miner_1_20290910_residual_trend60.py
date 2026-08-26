import numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-09-10'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=np.log(P).diff()
raw=np.log(P.shift(5)/P.shift(65)); res=raw.sub(raw.median(axis=1),axis=0)
vol=R.rolling(60,min_periods=40).std().shift(1); sig=(res/vol).replace([np.inf,-np.inf],np.nan)
fwd=np.log(P.shift(-10)/P); rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for a,b in [('full',r),('2020_2024',r.loc[:'2024-12-31']),('2025_2026',r.loc['2025-01-01':'2026-12-31']),('2027_2028',r.loc['2027-01-01':'2028-12-31']),('recent',r.loc['2028-09-01':])]:
 if len(b): print(a,'dates',len(b),'avg_n',round(b.n.mean(),2),'coverage',round(b.n.mean()/15,4),'IC',round(b.ic.mean(),6),'ICIR',round(b.ic.mean()/b.ic.std(ddof=1),6),'hit',round((b.ic>0).mean(),4))
rank=sig.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'assets',15,'data_dates',len(P))
for h in [5,10,20,40]:
 yy=np.log(P.shift(-h)/P); vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,'dates',len(vals),'IC',round(np.nanmean(vals),6),'ICIR',round(np.nanmean(vals)/np.nanstd(vals,ddof=1),6))
sig.stack().rename('signal').to_csv('scripts/miner_1_20290910_residual_trend60_signal.csv')
r.reset_index().to_csv('scripts/miner_1_20290910_residual_trend60_ic.csv',index=False)
