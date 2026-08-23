import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-03-05'); base='../persistent/stock_data/'
P=pd.DataFrame({s:pd.read_csv(base+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
R=P.pct_change(); r40=P.pct_change(40); vol=R.rolling(20,min_periods=15).std()*np.sqrt(252)
breadth=(r40>0).mean(axis=1)
f=(r40/vol).mul((0.5+breadth).shift(1),axis=0)
f.to_csv('scripts/miner_2_20310306_breadth_gated_trend_signal.csv',index_label='date')
for h in [5,10,20]:
 rows=[]
 for i in range(len(P)-h):
  x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: rows.append((P.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=a.ic
 print('H',h,'dates',len(q),'meanN',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,6),'IC',round(q.mean(),8),'ICIR',round(q.mean()/q.std(ddof=1),8),'hit',round((q>0).mean(),6),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[a.index].mean(),6))
 for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2031-03-05')]:
  z=q.loc[lo:hi]
  if len(z): print('REG',lo,round(z.mean(),8),len(z))
print('period',P.index.min().date(),P.index.max().date())
