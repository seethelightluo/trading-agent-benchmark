import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d): D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
v20=r.rolling(20).std()*np.sqrt(252); v60=r.rolling(60).std()*np.sqrt(252); mom20=p.pct_change(20)
f=((v20/v60-1.0) - 0.25*(mom20/(v60+0.05))).shift(1)
f.to_csv('scripts/miner_2_20330915_volatility_compression_signal.csv',index_label='date')
for h in [10,20,40,60]:
 ic=[]
 for dt in p.index:
  x=f.loc[dt]; y=p.pct_change(h).shift(-h).loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(ic); a=a[np.isfinite(a)]
 print(h,'IC',a.mean(),'ICIR',a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),'hit',(a>0).mean(),'dates',len(a))
valid=f.notna().sum(axis=1); print('coverage',valid.sum()/(len(f)*len(U)),'avgN',valid.mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('range',p.index.min(),p.index.max())
