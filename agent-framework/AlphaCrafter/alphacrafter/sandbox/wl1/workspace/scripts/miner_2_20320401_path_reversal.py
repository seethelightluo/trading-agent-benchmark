import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}; cut=pd.Timestamp('2032-03-31')
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date'); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); net=r.rolling(40,min_periods=30).sum(); path=r.abs().rolling(40,min_periods=30).sum(); f=(-net/(path+1e-9)).shift(1); fr={h:np.log(px.shift(-h)/px) for h in [1,5,10,20]}
def run(h):
 v=[];n=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));n.append(len(z))
 s=pd.Series(v);return len(s),np.mean(n),s.mean(),s.mean()/s.std(),np.mean(s>0)
for h in [1,5,10,20]:print(h,run(h))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20320401_path_reversal_signal.csv',index=False);print('artifact',len(out),'coverage',len(out)/(len(f.index)*len(px.columns)))
