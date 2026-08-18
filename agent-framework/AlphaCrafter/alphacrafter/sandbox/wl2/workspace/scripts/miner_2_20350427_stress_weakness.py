import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 d=d.copy();d.date=pd.to_datetime(d.date);return d.drop_duplicates('date').set_index('date').close.astype(float)
P=pd.concat({s:load(s) for s in U},axis=1).sort_index();r=P.pct_change();v=get_index_daily_data('VIX',5000);v=v.copy();v.date=pd.to_datetime(v.date);v=v.drop_duplicates('date').set_index('date').close.astype(float).reindex(P.index).ffill();breadth=(r>0).mean(axis=1).rolling(20,min_periods=15).mean();stress=((v>v.rolling(120,min_periods=60).median())|(breadth<.4)).shift(1);vol=r.rolling(40,min_periods=20).std()*np.sqrt(252);f=(-(P.pct_change(20)/vol).sub((P.pct_change(20)/vol).median(axis=1),axis=0)).shift(1).where(stress,0.)
f.to_csv('../persistent/miner_2_20350427_stress_relative_weakness_reversal_signal.csv',index_label='date')
for h in [5,10,20,40]:
 a=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append((P.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(a,columns=['date','ic','n']).set_index('date');print(h,len(q),q.n.mean(),q.ic.mean(),q.ic.mean()/q.ic.std()*np.sqrt(252),(q.ic>0).mean())
print('coverage',f.notna().mean().mean())
