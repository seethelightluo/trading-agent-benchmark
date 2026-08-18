import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
p={a:pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in A}; r=pd.DataFrame({a:x.pct_change() for a,x in p.items()}).sort_index(); r=r.loc[r.index <= pd.Timestamp('2026-07-15')]
# residual momentum: 20d asset return minus rolling cross-asset median return, volatility normalized
m=r.median(axis=1); rm=r.rolling(20,min_periods=15).sum().sub(m.rolling(20,min_periods=15).sum(),axis=0); vol=r.rolling(20,min_periods=15).std(); f=rm/vol
ics=[]
for i,d in enumerate(r.index[:-1]):
 z=pd.concat([f.loc[d],r.iloc[i+1]],axis=1).dropna()
 if len(z)>=8: ics.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
s=pd.Series([x[1] for x in ics],index=[x[0] for x in ics]); print('dates',len(s),'avg_names',np.mean([x[2] for x in ics]),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean(),'coverage',np.mean([x[2] for x in ics])/15)
for h in [5,10]:
 q=[]
 for i,d in enumerate(r.index[:-h]):
  z=pd.concat([f.loc[d],r.iloc[i+1:i+h+1].sum()],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q);print(h,q.mean(),q.mean()/q.std(),len(q))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=s[(s.index.year>=lo)&(s.index.year<=hi)];print(lo,len(q),q.mean(),q.mean()/q.std())
out=f;out.to_csv('scripts/miner_2_20260730_residual_mom_signal.csv',index_label='date')
