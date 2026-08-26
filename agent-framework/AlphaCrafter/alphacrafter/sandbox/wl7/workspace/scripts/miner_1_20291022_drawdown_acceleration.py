import numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-10-20'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
# Drawdown acceleration: compare short-window drawdown with medium-window drawdown.
# Positive values identify assets recovering from a medium-term peak; lagged one day.
dd20=P/P.rolling(20,min_periods=15).max()-1
dd60=P/P.rolling(60,min_periods=40).max()-1
vol20=np.log(P).diff().rolling(20,min_periods=15).std()
raw=(dd20-dd60).div(vol20)
sig=raw.shift(1).replace([np.inf,-np.inf],np.nan)
Y={h:np.log(P.shift(-h)/P) for h in [5,10,20,40]}
for orient in [1,-1]:
 s=sig*orient; print('ORIENTATION',orient)
 for h,y in Y.items():
  rows=[]
  for dt in s.index:
   z=pd.concat([s.loc[dt],y.loc[dt]],axis=1).dropna()
   if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
  r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
  print('h',h,'dates',len(r),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.mean()/15,4),'IC',round(r.ic.mean(),6),'ICIR',round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4))
  for label,sub in [('2025_26',r.loc['2025':'2026']),('2027_28',r.loc['2027':'2028']),('recent',r.loc['2028-10-01':])]:
   if len(sub): print(label,len(sub),round(sub.ic.mean(),6),round(sub.ic.mean()/sub.ic.std(ddof=1),6))
print('dates',len(P),'avg signal coverage',round(sig.notna().sum(axis=1).mean()/15,4))
sig.stack().rename('signal').to_csv('scripts/miner_1_20291022_drawdown_acceleration_signal.csv')
y=Y[10]; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20291022_drawdown_acceleration_ic.csv',index=False)
