import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,5200)
 if d is None or len(d)==0:d=get_index_daily_data(s,5200)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=np.log(px).diff();
ret=r.rolling(60,min_periods=45).sum(); down=r.where(r<0).rolling(40,min_periods=25).std()+1e-9
bread=(r>0).rolling(20,min_periods=15).mean()
f=((ret/down)*(0.5+0.5*bread)).shift(1); f=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1)+1e-9,axis=0)
fw={h:np.log(px.shift(-h)/px) for h in [5,10,20,40]}; rows=[];turn=[];prev=None
for dt in f.index:
 a=f.loc[dt];y=fw[10].loc[dt];ok=a.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(y[ok]),ok.sum()))
  if prev is not None:
   q=ok&prev.notna();turn.append((a[q].rank(pct=True)-prev[q].rank(pct=True)).abs().mean())
  prev=a
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');x=z.ic
print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15);print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(x.mean(),x.mean()/x.std(),(x>0).mean(),np.nanmean(turn)))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=x[(x.index>=lo)&(x.index<=hi+'-12-31')];print(lo,hi,len(q),q.mean(),q.mean()/q.std(),(q>0).mean())
for h in [5,10,20,40]:
 a=[]
 for dt in f.index:
  q=f.loc[dt];y=fw[h].loc[dt];ok=q.notna()&y.notna()
  if ok.sum()>=8:a.append(q[ok].corr(y[ok]))
 print('decay',h,np.mean(a),len(a))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20340623_persistent_trend_signal.csv',index=False)
