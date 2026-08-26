import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}; cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill();r=cl.pct_change()
cs=r.rolling(60,min_periods=40).sum();res=cs-cs.median(axis=1).values[:,None]; rec=r.rolling(10,min_periods=7).sum();v=r.rolling(20,min_periods=12).std();bread=(r>0).rolling(20,min_periods=12).mean().mean(axis=1);weak=bread<bread.rolling(120,min_periods=60).quantile(.55);sig=((-res/(v*np.sqrt(20)+.02)+1.5*rec/(v*np.sqrt(20)+.02)).where(weak)).shift(1)
def run(start):
 f=cl.shift(-40)/cl-1; z=[]
 for i in range(len(sig)):
  if sig.index[i]<pd.Timestamp(start):continue
  x=sig.iloc[i];y=f.iloc[i];ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))
 z=pd.Series(z).dropna();return len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
for s in ['2020-01-01','2028-01-01','2030-01-01','2032-01-01','2033-01-01','2034-01-01']:print(s,run(s))
