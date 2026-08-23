import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100:d=get_index_daily_data(s,days=3000)
 if d is not None and len(d):px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); ret20=p/p.shift(20)-1
res=ret20-ret20.median(axis=1).values[:,None]; vol=r.rolling(40).std()*np.sqrt(20); breadth=ret20.gt(0).mean(axis=1)
confirm=(.5+abs(breadth-.5)*2)*(.75+.5*(breadth>.5).astype(float))
fac=(-res.div(vol+1e-12).mul(confirm,axis=0)).shift(1)
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('cutoff',o.index.max(),'dates',len(o),'avg_n',o.n.mean(),'assets',len(U),'coverage',fac.notna().sum().sum()/(len(fac)*len(U)))
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(o.ic.mean(),o.ic.mean()/(o.ic.std(ddof=1)+1e-12),(o.ic>0).mean(),fac.rank(axis=1).diff().abs().mean().mean()/len(U)))
for h in [2,5,10,20]:
 yy=p.shift(-h)/p-1;v=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(v),len(v))
for name,sl in [('2025-27',o.loc['2025':'2027']),('online',o.loc['2026-07-16':'2027-04-07'])]:print(name,len(sl),sl.ic.mean(),sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12))
fac.to_csv('scripts/miner_1_20270408_residual_reversal_signal.csv')
