import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)==0:d=get_index_daily_data(s,3000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=np.log(px).diff(); m=r.mean(axis=1)
# residual short-horizon reversal: remove common cross-asset move, scale by idiosyncratic volatility
beta=r.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var()+1e-9,axis=0)
res=r.sub(beta.mul(m,axis=0)); v=res.rolling(20,min_periods=15).std()+1e-9
f=(-res.rolling(5,min_periods=5).sum()/v).shift(1)
fw={h:np.log(px.shift(-h)/px) for h in [5,10,20,40]}; rows=[]; ts=[]; prev=None
for dt in f.index:
 a=f.loc[dt]; y=fw[10].loc[dt]; ok=a.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(y[ok]),ok.sum()))
  if prev is not None:
   q=ok&prev.notna();ts.append((a[q].rank(pct=True)-prev[q].rank(pct=True)).abs().mean())
  prev=a
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');x=z.ic.dropna();print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15);print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(x.mean(),x.mean()/x.std(),(x>0).mean(),np.nanmean(ts)))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=x[(x.index>=lo)&(x.index<=hi+'-12-31')];print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std())
for h in [5,10,20,40]:
 a=[]
 for dt in f.index:
  q=f.loc[dt];y=fw[h].loc[dt];ok=q.notna()&y.notna()
  if ok.sum()>=8:a.append(q[ok].corr(y[ok]))
 print('decay',h,np.mean(a),np.mean(a)/np.std(a),len(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20340707_residual_reversal_signal.csv',index=False)
