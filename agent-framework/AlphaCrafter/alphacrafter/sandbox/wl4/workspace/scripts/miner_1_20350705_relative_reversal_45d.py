import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat([pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in S],axis=1).sort_index()
r=p.pct_change(); raw=r.rolling(45,min_periods=30).sum(); common=raw.median(axis=1)
vol=r.rolling(45,min_periods=30).std(); sig=(-(raw.sub(common,axis=0))/vol.replace(0,np.nan)).shift(1)
# signal is known at decision date; forward return starts after signal date
f=p.shift(-10)/p-1
rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=relative_reversal_45d dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15))
print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
for k in [120,260,520,1040]:
 q=a.tail(k); print('recent',k,'IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
print('turnover_proxy',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; zics=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:zics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'IC %.8f ICIR %.8f'%(np.nanmean(zics),np.nanmean(zics)/np.nanstd(zics,ddof=1)))
os.makedirs('scripts/artifacts',exist_ok=True);sig.to_csv('scripts/artifacts/miner_1_20350705_relative_reversal_45d_signal.csv');a.to_csv('scripts/artifacts/miner_1_20350705_relative_reversal_45d_ic.csv')
