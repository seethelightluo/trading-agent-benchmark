import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in u:
 x=get_stock_daily_data(s,days=2200)
 if x is None or len(x)<80: x=get_index_daily_data(s,days=2200)
 if x is not None and len(x): D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Directional persistence: return over 15d weighted by fraction of positive sessions,
# divided by total realized volatility. This rewards steady trends and penalizes choppy moves.
ret=p/p.shift(15)-1; vol=r.rolling(20).std(); pos=(r>0).rolling(15).mean()
f=ret*(0.5+pos)/(vol+1e-6)
rows=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8:
  z.f=z.f.clip(z.f.quantile(.05),z.f.quantile(.95)); rows.append((p.index[i],len(z),*[z.f.corr(r.iloc[i+1:i+1+h].sum()) for h in [1,3,5,10]]))
a=pd.DataFrame(rows,columns=['date','n','ic1','ic3','ic5','ic10']).set_index('date')
for c in ['ic1','ic3','ic5','ic10']:
 x=a[c].dropna(); print(c,'dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'period',a.index.min(),a.index.max())
for name,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-27',a.index>='2026-01-01')]:
 x=a.loc[mask,'ic1'].dropna(); print(name,len(x),x.mean(),x.mean()/x.std(ddof=1))
