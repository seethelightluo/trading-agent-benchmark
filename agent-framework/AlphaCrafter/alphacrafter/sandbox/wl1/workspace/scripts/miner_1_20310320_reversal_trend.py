import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; q={}
for s in U:
 try:d=get_index_daily_data(s,days=4100)
 except Exception:d=get_stock_daily_data(s,days=4100)
 if d is not None and len(d): d=d.copy();d.date=pd.to_datetime(d.date);q[s]=d.set_index('date').close.astype(float).sort_index()
p=pd.DataFrame(q).sort_index(); r=np.log(p).diff(); v=r.rolling(20).std();
# short reversal conditioned on distance from slow trend; scale reversal by recent volatility
raw=-(p/p.shift(5)-1)/(v*np.sqrt(5)+1e-9) + .30*(p/p.shift(60)-1)/(r.rolling(60).std()*np.sqrt(60)+1e-9)
f=raw.shift(1).rank(axis=1,pct=True)
def go(h):
 y=p.shift(-h)/p-1; a=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(d)
 return pd.Series(a,index=ds).dropna()
print('instruments',len(q),'dates',len(p))
for h in [1,5,10,20]:
 x=go(h);print(h,'IC %.6f ICIR %.6f hit %.4f n=%d'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),len(x)))
x=go(1)
for y,g in x.groupby(x.index.year):print(y,'%.6f n=%d'%(g.mean(),len(g)))
print('turnover',f.diff().abs().mean(axis=1).dropna().mean())
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20310320_reversal_trend_signal.csv',index=False)
