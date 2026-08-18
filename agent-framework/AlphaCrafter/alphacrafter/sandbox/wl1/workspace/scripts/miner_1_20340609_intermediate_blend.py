import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,5200)
 if d is None or len(d)==0:d=get_index_daily_data(s,5200)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill();r=np.log(px).diff();vol=r.rolling(20,min_periods=15).std()+1e-9
# Intermediate-horizon trend/reversal blend: 20d risk-adjusted momentum plus 5d reversal.
# Lag one completed day and rank cross-sectionally.
m=(r.rolling(20,min_periods=18).sum()/vol).shift(1); q=(-r.rolling(5,min_periods=5).sum()/vol).shift(1)
def cs(x):return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1)+1e-9,axis=0)
f=cs(m)+0.35*cs(q); fw={h:np.log(px.shift(-h)/px) for h in [5,10,20,40]}
rows=[];turn=[];prev=None
for dt in f.index:
 a=f.loc[dt];y=fw[10].loc[dt];ok=a.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(y[ok]),ok.sum()))
  if prev is not None:
   z=ok&prev.notna();turn.append((a[z].rank(pct=True)-prev[z].rank(pct=True)).abs().mean())
  prev=a
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');x=z.ic.dropna();print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15);print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(x.mean(),x.mean()/x.std(),(x>0).mean(),np.nanmean(turn)))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 a=x[(x.index>=lo)&(x.index<=hi+'-12-31')];print(lo,hi,len(a),a.mean(),a.mean()/a.std() if a.std()>0 else np.nan,(a>0).mean())
for h in [5,10,20,40]:
 a=[]
 for dt in f.index:
  u=f.loc[dt];y=fw[h].loc[dt];ok=u.notna()&y.notna()
  if ok.sum()>=8:a.append(u[ok].corr(y[ok]))
 print('decay',h,np.nanmean(a),len(a))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20340609_intermediate_blend_signal.csv',index=False)
