import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2028-01-27'); px={}
for s in U:
 d=get_stock_daily_data(s,days=3000); x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); px[s]=x[x.date<=cut].drop_duplicates('date').set_index('date').close
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change(); vol=r.rolling(20,min_periods=20).std(); raw=-(r.rolling(3,min_periods=3).sum().sub(r.rolling(3,min_periods=3).sum().median(axis=1),axis=0))/(vol+1e-8); Y=P.shift(-10)/P-1
rows=[]
for dt in raw.index:
 z=pd.concat([raw.loc[dt],Y.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); rank=raw.rank(axis=1,pct=True)
print('cutoff',P.index.max().date(),'dates',len(R),'instruments',len(U),'mean_n',R.n.mean(),'coverage',R.n.mean()/15)
print('IC',R.ic.mean(),'ICIR',R.ic.mean()/R.ic.std(ddof=1),'hit',(R.ic>0).mean(),'turn',rank.diff().abs().mean(axis=1).mean())
for h in [5,10,20]:
 q=[]; YY=P.shift(-h)/P-1
 for dt in raw.index:
  z=pd.concat([raw.loc[dt],YY.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(q),len(q))
for name,sub in [('2020-22',R.loc['2020':'2022']),('2023-25',R.loc['2023':'2025']),('2026-27',R.loc['2026':'2027']),('recent',R.tail(60))]: print(name,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1) if len(sub)>1 else np.nan)
raw.to_csv('scripts/miner_1_20280128_volscaled_relative_reversal_signal.csv')
