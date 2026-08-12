import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None and len(d): D[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index(); r=np.log(px/px.shift(1))
# Inefficient-move reversal: reverse recent return, emphasized when path is choppy.
# Efficiency = absolute net 20d move / sum absolute daily moves; shift avoids look-ahead.
eff=r.abs().rolling(20).sum().replace(0,np.nan)
eff=(r.rolling(20).sum().abs()/eff).clip(0,1)
f=(-(r.rolling(10).sum())*(1-eff)).shift(1)
f=f.sub(f.median(axis=1),axis=0)
rows=[]
for i,dt in enumerate(px.index):
 if i+5>=len(px): break
 z=pd.concat([f.loc[dt],np.log(px.iloc[i+5]/px.iloc[i])],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),z.iloc[:,0].corr(z.iloc[:,1],method='pearson')))
x=pd.DataFrame(rows,columns=['date','n','ic','pic']).set_index('date')
print('dates',len(px),'instruments',len(D),'ICobs',len(x),'avgN',x.n.mean(),'minN',x.n.min())
print('spearman IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
print('pearson IC %.6f ICIR %.6f'%(x.pic.mean(),x.pic.mean()/x.pic.std()))
for label,z in [('recent250',x.tail(250)),('2020_22',x.loc[:'2022-12-31']),('2023_25',x.loc['2023-01-01':'2025-12-31']),('2026_29',x.loc['2026-01-01':])]:
 print(label,len(z),'ic',z.ic.mean(),'icir',z.ic.mean()/z.ic.std() if len(z)>1 else np.nan,'hit',(z.ic>0).mean())
rr=f.rank(axis=1,pct=True); common=rr.index.intersection(x.index)
print('coverage',f.notna().sum().sum()/(len(f)*len(D)),'turnover_proxy',rr.loc[common].diff().abs().mean(axis=1).mean())
out=f.loc[x.index].stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20290726_inefficient_reversal10_signal.csv',index=False);print('artifact rows',len(out))
