import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={};V={}
for s in U:
 x=get_stock_daily_data(s,6000)
 if x is not None:
  x=x.copy(); x.date=pd.to_datetime(x.date); x=x.set_index('date'); D[s]=x.close.astype(float); V[s]=pd.to_numeric(x.volume,errors='coerce').replace(0,np.nan)
p=pd.DataFrame(D).sort_index().ffill(); vol=pd.DataFrame(V).reindex(p.index).ffill(); r20=p.pct_change(20); r60=p.pct_change(60)
acc=(r20-r20.median(axis=1).values[:,None])-(r60-r60.median(axis=1).values[:,None])
vs=np.log(vol/vol.rolling(60,min_periods=20).median()).replace([np.inf,-np.inf],np.nan); vc=vs.sub(vs.median(axis=1),axis=0).clip(-1.5,1.5).fillna(0)
path=(p.pct_change(40).abs()/(p.pct_change().abs().rolling(40,min_periods=25).sum())).clip(0,1)
path=path.div(path.median(axis=1),axis=0).clip(.5,1.5).fillna(1)
sig=(acc*(1+.25*vc)*path).shift(1)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],p.shift(-10).loc[dt]/p.loc[dt]-1],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15,'IC10',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 z=q.loc[a:b,'ic']; print('regime',a,b,len(z),z.mean(),z.mean()/z.std())
q.to_csv('scripts/miner_1_20351109_path_volume_accel_ic.csv'); sig.to_csv('scripts/miner_1_20351109_path_volume_accel_signal.csv',index_label='date')
