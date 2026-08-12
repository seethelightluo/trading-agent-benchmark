import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):D[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index();r=np.log(px/px.shift(1))
# Reversal after downside-dominated selloffs: prior 5d loss scaled by downside-vol share.
down=(r.clip(upper=0)**2).rolling(20).mean(); total=(r**2).rolling(20).mean(); share=(down/total.replace(0,np.nan)).clip(0,1)
f=(-(r.rolling(5).sum())*(0.5+share)).shift(1);f=f.sub(f.median(axis=1),axis=0)
rows=[]
for i,dt in enumerate(px.index):
 if i+5>=len(px):break
 z=pd.concat([f.loc[dt],np.log(px.iloc[i+5]/px.iloc[i])],axis=1).dropna()
 if len(z)>=8:rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),z.iloc[:,0].corr(z.iloc[:,1],method='pearson')))
x=pd.DataFrame(rows,columns=['date','n','ic','pic']).set_index('date')
print('dates',len(px),'instruments',len(D),'ICobs',len(x),'avgN',x.n.mean(),'minN',x.n.min())
print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()));print('pearson',x.pic.mean(),x.pic.mean()/x.pic.std())
for label,z in [('recent250',x.tail(250)),('2026_29',x.loc['2026-01-01':])]:print(label,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean())
rr=f.rank(axis=1,pct=True);print('coverage',f.notna().sum().sum()/(len(f)*len(D)),'turnover_proxy',rr.loc[x.index].diff().abs().mean(axis=1).mean())
out=f.loc[x.index].stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20290726_downside_reversal5_signal.csv',index=False)
