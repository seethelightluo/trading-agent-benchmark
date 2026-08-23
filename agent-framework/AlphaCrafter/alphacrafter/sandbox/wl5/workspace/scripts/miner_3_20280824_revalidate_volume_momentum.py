import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close={}; vol={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  z=d.set_index('date'); close[s]=z['close']; vol[s]=z['volume']
p=pd.DataFrame(close).sort_index().ffill(); v=pd.DataFrame(vol).reindex(p.index).ffill(); r=p.pct_change()
ret20=r.rolling(20).sum(); risk=r.rolling(20).std()*np.sqrt(20); shock=(v.rolling(5).mean()/(v.rolling(60).mean()+1e-12)-1).clip(-1,3)
f=(ret20/(risk+1e-6))*(1+0.35*shock)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(y[ok]),ok.mean()))
d=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for n,z in [('full',d),('recent_1y',d[d.date>='2027-08-24']),('recent_6m',d[d.date>='2028-02-01']),('online',d[d.date>='2028-07-16'])]:
 if len(z): print(n,'dates',len(z),'avg_names',round(z.coverage.mean()*15,2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(d.coverage.mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'range',d.date.min(),d.date.max(),'instruments',len(close))
