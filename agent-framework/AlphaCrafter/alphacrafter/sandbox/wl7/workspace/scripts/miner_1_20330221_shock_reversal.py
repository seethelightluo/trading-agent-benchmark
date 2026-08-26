import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date')['close'].rename(a) for a in A],axis=1).dropna(); R=P.pct_change()
# shock reversal: 5d return ending t-4, scaled by recent 20d volatility ending t-4; captures delayed overreaction
F=-(P.shift(4)/P.shift(9)-1)/(R.rolling(20).std().shift(4)+1e-12)
for h in [5,10,20]:
 z=[]
 for j in range(len(P)-h):
  q=pd.concat([F.iloc[j],(P.iloc[j+h]/P.iloc[j]-1).rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.y).statistic)
 z=np.array(z);print(h,len(z),np.mean(z),np.mean(z)/(np.std(z,ddof=1)+1e-12),np.mean(z>0))
r=F.rank(axis=1,pct=True);t=[]
for j in range(1,len(r)):
 q=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(q)>=8:t.append(np.mean(abs(q.iloc[:,0]-q.iloc[:,1])))
print('coverage',F.notna().mean().mean(),'dates',len(F),'turn',np.mean(t))
F.to_csv('scripts/miner_1_20330221_shock_reversal_signal.csv')
