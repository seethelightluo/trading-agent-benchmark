import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat([pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in S],axis=1).sort_index(); r=p.pct_change()
# acceleration reversal: reverse recent 20d return after removing slower 60d trend, risk scaled
v=r.rolling(20,min_periods=15).std(); sig=(-(p.pct_change(20)-p.pct_change(60)/3)/v).shift(1); f=p.shift(-10)/p-1
out=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
 if len(z)>=8: out.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); print('candidate=acceleration_reversal'); print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(15*len(a))); print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
for k in [120,260,520,1040]:
 q=a.tail(k); print('recent',k,'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('turnover',sig.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 f2=p.shift(-h)/p-1; q=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f2.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(q),np.mean(q)/np.std(q,ddof=1))
os.makedirs('scripts/artifacts',exist_ok=True); sig.to_csv('scripts/artifacts/miner_2_20350315_acceleration_reversal_signal.csv'); a.to_csv('scripts/artifacts/miner_2_20350315_acceleration_reversal_ic.csv')
