import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{B}/{s}.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index()
# Short-term reversal, volatility scaled and cross-sectionally centered; lag ensures no lookahead.
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); f=(-r.rolling(3).sum()/vol).shift(1); f=f.sub(f.mean(axis=1),axis=0)
for h in [5,10,20,40]:
 out=[]
 for i,d in enumerate(P.index[:-h]):
  z=pd.concat([f.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: out.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5),'hit',round((q.ic>0).mean(),4))
 if h==10:
  for a,b in [('2020','2024'),('2025','2029'),('2030','2035')]:
   z=q.loc[a:b];print('REG',a,b,len(z),round(z.ic.mean(),5),round(z.ic.mean()/z.ic.std(ddof=1),5))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5)); f.to_csv('scripts/miner_1_20350202_short_reversal_signal.csv',index_label='date')
