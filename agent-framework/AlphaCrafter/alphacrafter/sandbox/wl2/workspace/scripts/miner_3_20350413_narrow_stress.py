import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 d=d.copy(); d.date=pd.to_datetime(d.date)
 return d.drop_duplicates('date').set_index('date').close.astype(float)
P=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=P.pct_change()
v=get_index_daily_data('VIX',5000); v=v.copy(); v.date=pd.to_datetime(v.date)
v=v.drop_duplicates('date').set_index('date').close.astype(float).reindex(P.index).ffill()
breadth=(r>0).mean(axis=1).rolling(20,min_periods=15).mean()
# Narrow, persistent stress: both macro and cross-asset weakness, all lagged one day.
stress=((v>v.rolling(120,min_periods=60).median()) & (breadth<0.40)).shift(1)
loc=(P-P.rolling(120,min_periods=80).min())/(P.rolling(120,min_periods=80).max()-P.rolling(120,min_periods=80).min())
f=(-loc).shift(1).where(stress,0.0)
f.to_csv('../persistent/miner_3_20350413_narrow_stress_location_signal.csv',index_label='date')
for h in [5,10,20,40]:
 vals=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append((P.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); sd=q.ic.std()
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(ic,6),'ICIR',round(ic/sd*np.sqrt(252),6),'hit',round((q.ic>0).mean(),4))
 print('regimes',[(a,round(q.loc[a].ic.mean(),6),len(q.loc[a])) for a in [q.index<'2023-01-01',(q.index>='2023-01-01')&(q.index<'2029-01-01'),q.index>='2029-01-01']])
print('universe',len(U),'dates',len(P),'stress_rate',round(float(stress.mean()),4),'coverage',round(float(f.notna().mean().mean()),4),'active',round(float((f.abs().sum(axis=1)>0).mean()),4))
print('turnover_proxy',round(float(f.diff().abs().sum(axis=1).gt(0).mean()),4))
