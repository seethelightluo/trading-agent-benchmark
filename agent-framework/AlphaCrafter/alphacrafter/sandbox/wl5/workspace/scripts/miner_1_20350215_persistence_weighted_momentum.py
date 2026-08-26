import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data'
px={s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'].astype(float) for s in U}
close=pd.DataFrame(px).sort_index(); ret=close.pct_change()
# Momentum weighted by directional persistence: medium trend times excess fraction of up days.
mom=close.pct_change(40)
persist=ret.gt(0).rolling(40,min_periods=30).mean()-0.5
vol=ret.rolling(40,min_periods=30).std()
f=(mom*persist/(vol*np.sqrt(40))).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(close)-20):
 if close.index[i] < pd.Timestamp('2023-06-30'): continue
 a=f.iloc[i]; b=close.iloc[i+10]/close.iloc[i]-1; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((close.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
out=pd.DataFrame(rows,columns=['date','ic','n']).dropna()
mean=out.ic.mean(); sd=out.ic.std(ddof=1)
rank=f.rank(axis=1,pct=True); turnover=(rank.diff().abs().mean(axis=1)/2).reindex(out.date).mean()
print('dates',len(out),'mean_n',out.n.mean(),'coverage',out.n.mean()/15)
print('ic10',mean,'icir',mean/sd,'hit',np.mean(out.ic>0),'turnover',turnover)
for h in [5,10,20]:
 z=[]
 for i in range(len(close)-h):
  if close.index[i]<pd.Timestamp('2023-06-30'): continue
  a=f.iloc[i]; b=close.iloc[i+h]/close.iloc[i]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.mean(z),len(z))
out.to_csv('scripts/miner_1_20350215_persistence_weighted_momentum_signal.csv',index=False)
