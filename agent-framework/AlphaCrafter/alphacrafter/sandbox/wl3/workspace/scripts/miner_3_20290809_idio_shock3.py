import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):D[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index();r=np.log(px/px.shift(1)); idio=r.sub(r.median(axis=1),axis=0)
# Idiosyncratic shock reversal: reverse a recent 3-day residual move, emphasizing unusually large shocks.
vol=idio.rolling(60).std().replace(0,np.nan)
shock=idio.rolling(3).sum(); f=(-shock*shock.abs()/(vol*np.sqrt(3))).shift(1); f=f.sub(f.median(axis=1),axis=0)
rows=[]
for i,dt in enumerate(px.index):
 if i+1>=len(px):break
 z=pd.concat([f.loc[dt],r.iloc[i+1]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(px),'instruments',len(D),'ICobs',len(x),'avgN',x.n.mean(),'minN',x.n.min())
print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for label,z in [('recent250',x.tail(250)),('2026_29',x.loc['2026-01-01':]),('2023_25',x.loc['2023-01-01':'2025-12-31']),('2026_27',x.loc['2026-01-01':'2027-12-31']),('2028_29',x.loc['2028-01-01':])]:
 print(label,len(z),('%.6f %.6f %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean())) if len(z) else '')
rr=f.rank(axis=1,pct=True);print('coverage',f.notna().sum().sum()/(len(f)*len(D)),'turnover_proxy',rr.loc[x.index].diff().abs().mean(axis=1).mean())
out=f.loc[x.index].stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20290809_idio_shock3_signal.csv',index=False)
