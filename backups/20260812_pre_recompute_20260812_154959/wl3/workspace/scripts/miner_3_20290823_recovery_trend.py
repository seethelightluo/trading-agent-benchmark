import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):D[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index();r=np.log(px/px.shift(1));
# Trend persistence factor: combine medium-term return with recovery from the
# 60-session low, normalized by realized volatility; lagged one day.
ret20=np.log(px/px.shift(20)); low60=px.rolling(60).min(); recovery=np.log(px/low60); vol20=r.rolling(20).std()*np.sqrt(20)
f=(0.65*ret20+0.35*recovery)/(vol20+1e-12);f=f.shift(1);f=f.sub(f.median(axis=1),axis=0)
rows=[]
for i,dt in enumerate(px.index):
 if i+5>=len(px):break
 z=pd.concat([f.loc[dt],np.log(px.iloc[i+1]/px.iloc[i])],axis=1).dropna()
 if len(z)>=8:rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(px),'instruments',len(D),'ICobs',len(x),'avgN',x.n.mean(),'minN',x.n.min())
print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for label,z in [('recent250',x.tail(250)),('2026_29',x.loc['2026-01-01':])]:print(label,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean())
rr=f.rank(axis=1,pct=True);print('coverage',f.notna().sum().sum()/(len(f)*len(D)),'turnover_proxy',rr.loc[x.index].diff().abs().mean(axis=1).mean())
out=f.loc[x.index].stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20290823_recovery_trend_signal.csv',index=False)
