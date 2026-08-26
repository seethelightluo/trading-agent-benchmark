import numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-10-08'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=np.log(P).diff()
# Recovery-adjusted trend: favor assets with positive medium-term return and shallow/currently improving drawdown.
# All rolling statistics are shifted one day before signal use.
ret60=np.log(P/P.shift(60)); dd60=P/P.rolling(60,min_periods=40).max()-1
recovery20=P/P.shift(20)-1
vol20=R.rolling(20,min_periods=15).std()
# interpretable score: trend + recovery, penalized by drawdown and volatility
raw=(0.50*ret60 + 0.30*recovery20 + 0.20*dd60).div(vol20)
sig=raw.shift(1).replace([np.inf,-np.inf],np.nan)
for h in [5,10,20,40]:
 yy=np.log(P.shift(-h)/P); rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 if len(r):
  print('h',h,'dates',len(r),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.mean()/15,4),'IC',round(r.ic.mean(),6),'ICIR',round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4))
  for label,sub in [('2025_26',r.loc['2025':'2026']),('2027_28',r.loc['2027':'2028']),('recent',r.loc['2028-10-01':])]:
   if len(sub): print(label,'dates',len(sub),'IC',round(sub.ic.mean(),6),'ICIR',round(sub.ic.mean()/sub.ic.std(ddof=1),6))
rank=sig.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'data_dates',len(P))
sig.stack().rename('signal').to_csv('scripts/miner_1_20291008_recovery_adjusted_trend_signal.csv')
rows=[]; yy=np.log(P.shift(-10)/P)
for dt in sig.index:
 z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20291008_recovery_adjusted_trend_ic.csv',index=False)
print('valid_dates',len(P),'avg_valid_names',round(sig.notna().sum(axis=1).mean(),2))
