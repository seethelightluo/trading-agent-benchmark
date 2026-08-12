import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
acct=get_account_dict(); uni=acct.get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in uni:
 x=get_stock_daily_data(s,days=2200)
 if x is None or len(x)<80: x=get_index_daily_data(s,days=2200)
 if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change()
# Smoothed 5-session shock rebound: reverse recent move, volatility-scaled, blended with slower 20d trend.
vol=r.rolling(20).std(); shock=-(px/px.shift(5)-1)/(vol*np.sqrt(5)+1e-8)
trend=px/px.shift(20)-1
f=(shock.rolling(3).mean()+0.20*trend/(vol*np.sqrt(20)+1e-8)).clip(-10,10)
rows=[]
for i in range(len(px)-10):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8:
  z.f=z.f.clip(z.f.quantile(.05),z.f.quantile(.95)); rows.append((px.index[i],len(z),z.f.corr(z.y),z.f.corr(r.iloc[i+1:i+4].sum()),z.f.corr(r.iloc[i+1:i+6].sum()),z.f.corr(r.iloc[i+1:i+11].sum())))
a=pd.DataFrame(rows,columns=['date','n','ic1','ic3','ic5','ic10']).set_index('date')
for c in ['ic1','ic3','ic5','ic10']:
 x=a[c].dropna(); print(c,'dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).dropna().mean(),'coverage',f.notna().mean().mean(),'period',a.index.min(),a.index.max())
for name,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',((a.index>='2023-01-01')&(a.index<'2026-01-01'))),('2026-27',a.index>='2026-01-01')]:
 x=a.loc[mask,'ic1'].dropna(); print(name,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
