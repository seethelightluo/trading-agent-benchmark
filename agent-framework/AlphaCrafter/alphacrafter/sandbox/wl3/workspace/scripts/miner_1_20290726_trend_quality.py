import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):D[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index(); r=np.log(px/px.shift(1));
# medium-term trend quality: 60d return divided by 20d realized risk, requiring positive path consistency; lagged.
ret=r.rolling(60).sum(); risk=r.rolling(20).std(); consistency=(r>0).rolling(60).mean()-0.5
f=(ret/risk*(1+consistency)).shift(1)
rows=[]
for i,dt in enumerate(px.index):
 if i+10>=len(px):continue
 z=pd.concat([f.loc[dt],np.log(px.iloc[i+10]/px.iloc[i])],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(px),'instruments',len(D),'ICobs',len(x),'avgN',x.n.mean())
print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for name,z in [('all',x),('recent250',x.tail(250)),('2020_22',x.loc[:'2022-12-31']),('2023_25',x.loc['2023-01-01':'2025-12-31']),('2026_29',x.loc['2026-01-01':])]:
 print(name,len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
rr=f.rank(axis=1,pct=True);print('coverage',f.notna().sum().sum()/(len(f)*len(D)),'turnover_proxy',rr.diff().abs().mean(axis=1).mean())
out=f.loc[x.index].stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20290726_trend_quality_signal.csv',index=False);print('artifact rows',len(out))
