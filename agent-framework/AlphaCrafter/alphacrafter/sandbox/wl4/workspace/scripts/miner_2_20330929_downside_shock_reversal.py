import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): P[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2033-09-28']; R=P.pct_change(); r10=P.pct_change(10)
down=R.where(R<0).rolling(30,min_periods=15).std(); allv=R.rolling(30,min_periods=20).std()
# Contrarian response to recent moves, scaled by downside-risk asymmetry; lagged.
F=(-r10*(down/allv)).shift(1)
for h in [1,5,10,20]:
 fut=P.shift(-h)/P-1; rows=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fut.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(rows,columns=['date','n','ic']); s=q.ic
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
 for k in [120,260,520]:
  x=s.tail(k);print(' recent',k,round(x.mean(),6),round(x.mean()/x.std(),6))
print('coverage',round(F.notna().sum(axis=1).mean()/15,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
F.to_csv('scripts/artifacts/miner_2_20330929_downside_shock_reversal_signal.csv')
