import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-02-19'); base='../persistent/stock_data/'
P=pd.DataFrame({s:pd.read_csv(base+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
R=P.pct_change(); residual=R.sub(R.mean(axis=1),axis=0)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
vp=v.shift(1).rolling(252,min_periods=100).rank(pct=True)
f=(-residual.rolling(5,min_periods=4).sum()).mul((0.75+0.5*vp),axis=0).shift(1)
f.to_csv('scripts/miner_2_20310220_vix_percentile_residual_reversal_signal.csv',index_label='date')
for h in [5,10,20]:
 rows=[]
 for i in range(len(P)-h):
  x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: rows.append((P.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=a.ic
 print('H',h,'dates',len(q),'meanN',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,6),'IC',round(q.mean(),8),'ICIR',round(q.mean()/q.std(ddof=1),8),'hit',round((q>0).mean(),6),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[a.index].mean(),6))
 if h==5:
  for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2031-02-19')]:
   z=q.loc[lo:hi]
   if len(z): print('REG',lo,round(z.mean(),8),len(z))
print('period',P.index.min().date(),P.index.max().date())
