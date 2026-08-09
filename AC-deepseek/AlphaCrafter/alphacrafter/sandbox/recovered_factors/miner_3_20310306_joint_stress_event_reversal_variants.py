import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv').set_index('date')['close'] for s in S}).sort_index(); r=p.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv').set_index('date')['close'].reindex(p.index).ffill().pct_change(); d=pd.read_csv('../persistent/index_data/DXY.csv').set_index('date')['close'].reindex(p.index).ffill().pct_change(); e=(((v-v.rolling(60).mean())/v.rolling(60).std()>1)&((d-d.rolling(60).mean())/d.rolling(60).std()>.25)).astype(float)
for L in [1,3,5,10]:
 # response over L sessions ending before signal date
 x=r.rolling(L).sum().shift(1); f=x.mul(e,axis=0).rolling(240).sum().div(e.rolling(240).sum(),axis=0); f[e.rolling(240).sum()<8]=np.nan; f=f.sub(f.mean(axis=1),axis=0)
 for h in [1,5,10,20]:
  z=[]
  for i in range(len(p)-h):
   q=pd.concat([f.iloc[i],(p.shift(-h)/p-1).iloc[i]],axis=1).dropna()
   if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
  z=np.array(z);print('L',L,'H',h,'n',len(z),'IC',z.mean(),'IR',z.mean()/z.std(ddof=1))
 print('coverage',f.notna().mean().mean())
