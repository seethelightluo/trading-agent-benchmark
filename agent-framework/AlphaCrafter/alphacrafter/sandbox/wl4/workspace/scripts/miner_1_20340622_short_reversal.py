import numpy as np,pandas as pd,os
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3600)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().ffill(); R=P.pct_change();
# short-horizon oversold rebound, lagged one completed day
F=(-P.pct_change(10)/(R.rolling(40).std()*np.sqrt(252)+1e-12)).shift(1)
rows=[]
for i in range(len(P)-10):
 q=pd.concat([F.iloc[i],(P.iloc[i+10]/P.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(q)>=8: rows.append((P.index[i],q.iloc[:,0].corr(q.y,method='spearman'),len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); os.makedirs('scripts/artifacts',exist_ok=True)
for k in [120,260,520,780,1200,len(a)]:
 b=a.tail(k);print(k,'dates',len(b),'avg_n',round(b.n.mean(),2),'IC',round(b.ic.mean(),6),'ICIR',round(b.ic.mean()/b.ic.std(ddof=1),6),'hit',round((b.ic>0).mean(),4))
print('TOTAL',len(a),a.index.min(),a.index.max(),'coverage',round(F.notna().sum(axis=1).mean()/15,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for st,en in [('2020','2023'),('2023','2027'),('2027','2031'),('2031','2034')]:
 b=a.loc[st:en];print('REGIME',st,en,len(b),round(b.ic.mean(),6),round(b.ic.mean()/b.ic.std(ddof=1),6))
F.to_csv('scripts/artifacts/miner_1_20340622_short_reversal_signal.csv');a.to_csv('scripts/artifacts/miner_1_20340622_short_reversal_ic.csv')
