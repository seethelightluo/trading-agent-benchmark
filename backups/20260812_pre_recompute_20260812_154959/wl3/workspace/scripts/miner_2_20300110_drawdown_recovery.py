import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>100: xs[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(xs).sort_index()
# 60d drawdown recovery: current close relative rolling high, normalized by 20d vol; contrarian (deep drawdown gets positive)
logr=np.log(p).diff(); vol=logr.rolling(20).std()
dd=p/p.rolling(60).max()-1
f=(-dd/(vol*np.sqrt(20))).shift(1)
# forward log returns
out=[]
for h in [1,3,5,10]:
 fr=np.log(p).shift(-h)-np.log(p)
 vals=[]; turns=[]; ninst=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ninst.append(len(a))
   turns.append((a.iloc[:,0].rank(pct=True)-0.5).abs().mean())
 z=pd.Series(vals).dropna()
 print(h,'dates',len(z),'avgN',round(np.mean(ninst),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(),6),'hit',round((z>0).mean(),4))
print('coverage',round(f.notna().sum().sum()/(f.size),4),'dates',len(p),'names',len(xs))
# artifact with values
f.stack().rename('signal').to_csv('scripts/miner_2_20300110_drawdown_recovery_signal.csv')
