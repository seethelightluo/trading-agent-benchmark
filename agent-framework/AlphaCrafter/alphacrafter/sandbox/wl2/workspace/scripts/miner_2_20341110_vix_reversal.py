import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 d=get_stock_daily_data(s,4500)
 if d is None or len(d)==0:d=get_index_daily_data(s,4500)
 d=d.copy();d.date=pd.to_datetime(d.date);return d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame({a:g(a) for a in A}).sort_index(); r=p.pct_change()
v=get_index_daily_data('VIX',4500);v=v.copy();v.date=pd.to_datetime(v.date);v=v.drop_duplicates('date').set_index('date').close.astype(float).reindex(p.index).ffill()
rv=r.rolling(20,min_periods=10).std()*np.sqrt(252); shock=-r.rolling(20,min_periods=10).sum()
active=v.shift(1)>v.shift(1).rolling(252,min_periods=126).median(); f=(shock/(rv+1e-8)).shift(1).where(active)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1]))
 a=pd.Series(z);print(h,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),(a>0).mean())
valid=f.notna().sum(1);ad=active&(valid>=8);print('dates',ad.sum(),'coverage',valid[ad].mean()/15,'turn',f.rank(pct=True,axis=1).diff().abs().mean(1).where(ad).mean())
f.loc[ad].to_csv('../persistent/miner_2_20341110_vix_reversal_signal.csv',index_label='date')
