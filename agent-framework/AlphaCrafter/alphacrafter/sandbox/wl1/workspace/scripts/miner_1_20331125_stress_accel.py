import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill()
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').iloc[:,0].astype(float).reindex(p.index).ffill()
# Stress-conditioned contrarian acceleration: reverse medium-term acceleration only in high-VIX regime
acc=(p.pct_change(20)-0.5*p.pct_change(60)/3).shift(1)
q=v.rolling(252,min_periods=126).quantile(.8).shift(1)
f=acc.where(v<=q,-acc)
fr=p.pct_change(10).shift(-10); rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');
print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'hit',(z.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for label,lo,hi in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033','2033-01-01','2033-10-26')]:
 qx=z.loc[lo:hi].ic;print(label,len(qx),qx.mean(),qx.mean()/qx.std())
for h in [5,10,20]:
 y=p.pct_change(h).shift(-h);a=[]
 for dt in f.index:
  b=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(b)>=8:a.append(b.iloc[:,0].corr(b.iloc[:,1]))
 print('decay',h,np.nanmean(a),np.nanmean(a)/np.nanstd(a))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20331125_stress_accel_signal.csv',index=False)
