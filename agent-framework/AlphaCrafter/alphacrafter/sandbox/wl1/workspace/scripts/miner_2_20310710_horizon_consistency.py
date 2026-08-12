import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=np.log(p).diff()
m10=np.log(p/p.shift(10)); m20=np.log(p/p.shift(20)); m40=np.log(p/p.shift(40)); rv=np.sqrt((r**2).rolling(40).mean())
sig=(.25*m10+.35*m20+.40*m40)/(rv+1e-8)*(1-((m10-m20).abs()+(m20-m40).abs())/(abs(m10)+abs(m20)+abs(m40)+1e-8)*.35)
rank=sig.rank(axis=1,pct=True)
for h in [1,5,10,20]:
 f=np.log(p.shift(-h)/p); xs=[]; ns=[]
 for dt in rank.index:
  a=pd.concat([rank.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8: xs.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a))
 x=np.array(xs); print(h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12),6),'hit',round(np.mean(x>0),4))
f=np.log(p.shift(-20)/p); xs=[]
for dt in rank.index:
 a=pd.concat([rank.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(a)>=8: xs.append((dt,a.iloc[:,0].corr(a.iloc[:,1],method='spearman')))
x=pd.DataFrame(xs,columns=['date','ic']).set_index('date')
for name,q in [('2020-22',('2020','2022')),('2023-25',('2023','2025')),('2026-28',('2026','2028')),('2029-30',('2029','2030')),('2031',('2031','2031'))]:
 z=x.loc[q[0]:q[1],'ic']; print(name,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('coverage',round(sig.notna().sum().sum()/sig.size,4),'dates',len(x),'assets',len(p.columns))
out=rank.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_2_20310710_horizon_consistency_signal.csv')
