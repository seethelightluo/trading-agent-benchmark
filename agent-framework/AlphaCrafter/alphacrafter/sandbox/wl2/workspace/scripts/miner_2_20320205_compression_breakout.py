import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<100:d=get_index_daily_data(s,days=3200)
 if d is not None:C[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index().ffill(); r=p.pct_change()
# Compression breakout: recent 5d directional return, amplified when short volatility is compressed vs long volatility.
vol5=r.rolling(5).std(); vol30=r.rolling(30).std(); f=(r.rolling(5).sum().div(vol5.replace(0,np.nan)))*(1-vol5.div(vol30.replace(0,np.nan)))
f=f.shift(1)
def ir(x):return x.mean()/x.std(ddof=1) if len(x)>1 and x.std(ddof=1)>0 else np.nan
def q(h):
 y=p.pct_change(h).shift(-h);o=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   c=z.f.corr(z.y)
   if np.isfinite(c):o.append((p.index[i],c,len(z)))
 return pd.DataFrame(o,columns=['date','ic','n']).set_index('date')
a=q(1);print('universe',len(C),'dates',len(a),'avg_n',round(a.n.mean(),3),'IC',round(a.ic.mean(),6),'ICIR',round(ir(a.ic),6),'hit',round((a.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True).diff().abs().mean().mean(),4))
for x,y in [('2020','2022'),('2023','2025'),('2026','2031'),('2030','2032')]:
 z=a.loc[x:y].ic;print('regime',x,y,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(ir(z),6) if len(z) else None)
for h in [3,5,10]:
 z=q(h);print('decay',h,'dates',len(z),'IC',round(z.ic.mean(),6) if len(z) else None)
f.to_csv('scripts/miner_2_20320205_compression_breakout_signal.csv')
