import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={};vv={}
for a in A:
 d=get_stock_daily_data(a,days=4000); z=d.set_index('date');px[a]=pd.to_numeric(z.close,errors='coerce');vv[a]=pd.to_numeric(z.volume,errors='coerce')
p=pd.concat(px,axis=1).sort_index();v=pd.concat(vv,axis=1).reindex(p.index); med=v.where(v>0).rolling(20,min_periods=10).median(); surprise=np.log(v.where(v>0)/med); f=(-p.pct_change(1)*surprise.clip(-1.5,1.5)).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10]:
 z=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:z.append(q.f.corr(q.y))
 x=np.array(z); print({'h':h,'dates':len(x),'IC':x.mean(),'ICIR':x.mean()/x.std(ddof=1),'hit':(x>0).mean()})
print('period',p.index.min().date(),p.index.max().date())
f.stack().rename('signal').to_csv('scripts/miner_2_20261217_volume_surprise_1d_signal.csv',header=True)
