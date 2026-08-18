import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in assets:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300: d=get_index_daily_data(s,days=6000)
 if d is not None and len(d): frames[s]=pd.Series(d.close.astype(float).values,index=pd.to_datetime(d.date)).groupby(level=0).last()
px=pd.concat(frames,axis=1).sort_index().ffill(limit=5); ret=px.pct_change()
r20=px/px.shift(20)-1; r60=px/px.shift(60)-1
# Slow robust risk estimate: rolling median of 20d realized vol over trailing 120 days, causal lag
rv20=ret.rolling(20).std(); vol120=rv20.rolling(120).median()*np.sqrt(252)
f=((r20-r60)/(vol120+1e-8)).shift(1); fwd=px.shift(-10)/px-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',ic.n.mean()/15,'IC10',ic.ic.mean(),'ICIR',ic.ic.mean()/ic.ic.std(),'hit',(ic.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=ic.loc[a:b]; print('regime',a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std())
for h in [5,10,20,40]:
 fw=px.shift(-h)/px-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(vals),len(vals))
out=[{'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(f.loc[dt,s])} for dt in f.index for s in assets if s in f and pd.notna(f.loc[dt,s])]
pd.DataFrame(out).to_csv('scripts/miner_1_20351207_slow_robust_accel_signal.csv',index=False)
ic.reset_index().to_csv('scripts/miner_1_20351207_slow_robust_accel_ic.csv',index=False)
