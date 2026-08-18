import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];xs={}
for s in A:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300:d=get_index_daily_data(s,days=6000)
 if d is not None and len(d):xs[s]=pd.Series(d.close.astype(float).values,index=pd.to_datetime(d.date)).groupby(level=0).last()
p=pd.concat(xs,axis=1).sort_index().ffill(limit=5);r=p.pct_change(); vol=r.rolling(20).std(); disp=pd.DataFrame(np.abs(r.values-np.nanmedian(r.values,axis=1)[:,None]),index=r.index,columns=r.columns).mean(axis=1).rolling(20).mean()
# short-horizon shock reversal, activated only when cross-asset shock dispersion is elevated
shock=r.rolling(3).sum(); f=(-shock/(vol+1e-8)).mul(disp/disp.rolling(120).median(),axis=0).shift(1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(p.shift(-10)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',ic.n.mean()/15,'IC10',ic.ic.mean(),'ICIR',ic.ic.mean()/ic.ic.std(),'hit',(ic.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=ic.loc[a:b];print('regime',a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std())
for h in [5,10,20,40]:
 fw=p.shift(-h)/p-1;v=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(v),len(v))
out=[{'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(f.loc[dt,s])} for dt in f.index for s in A if s in f and pd.notna(f.loc[dt,s])]
pd.DataFrame(out).to_csv('scripts/miner_1_20351221_dispersion_shock_reversal_signal.csv',index=False);ic.reset_index().to_csv('scripts/miner_1_20351221_dispersion_shock_reversal_ic.csv',index=False)
