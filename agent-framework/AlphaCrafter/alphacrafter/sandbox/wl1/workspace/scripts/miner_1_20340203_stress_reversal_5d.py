import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None: D[s]=d.set_index('date')['close'].rename(s)
p=pd.concat(D,axis=1).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std()*np.sqrt(252)
# 5-day residual reversal activated by broad realized-vol stress, with magnitude capped
stress=r.mean(axis=1).rolling(5).std().rolling(120).rank(pct=True)
f=(-(p/p.shift(5)-1)/vol).sub((-(p/p.shift(5)-1)/vol).mean(axis=1),axis=0)
f=f.mul((stress-.6).clip(lower=0),axis=0).shift(1)
def calc(k):
 rows=[]
 for i in range(len(p)-k):
  z=pd.concat([f.iloc[i],p.iloc[i+k]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0: rows.append((p.index[i],len(z),z.iloc[:,0].corr(z.iloc[:,1])))
 a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');return a
for k in [5,10,20]:
 a=calc(k);print('horizon',k,'dates',len(a),'avgN',a.n.mean(),'coverage',a.n.mean()/15,'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(),'hit',(a.ic>0).mean())
 for lo,hi in [(2020,2023),(2024,2026),(2027,2029),(2030,2032),(2033,2034)]:
  q=a[(a.index.year>=lo)&(a.index.year<=hi)];print((lo,hi),round(q.ic.mean(),5),round(q.ic.mean()/q.ic.std(),4))
a=calc(5);f.to_csv('scripts/miner_1_20340203_stress_reversal_5d_signal.csv')
print('turnover',f.rank(pct=True).diff().abs().mean().mean())
