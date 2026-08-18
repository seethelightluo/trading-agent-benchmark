import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,5200)
 if d is None or len(d)==0:d=get_index_daily_data(s,5200)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill();r=np.log(px).diff()
# Tail-resilient trend: medium return rewarded when positive-day breadth is high,
# penalized when downside-day magnitude dominates; all inputs lagged one day.
ret=r.rolling(30,min_periods=22).sum(); up=(r>0).rolling(30,min_periods=22).mean(); dn=(-r.clip(upper=0)).rolling(30,min_periods=22).mean(); upm=r.clip(lower=0).rolling(30,min_periods=22).mean()
raw=(ret/(r.rolling(30,min_periods=22).std()+1e-9))*(2*up-1)*(upm/(dn+1e-9)).clip(0.25,4).shift(1)
f=raw.sub(raw.mean(axis=1),axis=0).div(raw.std(axis=1)+1e-9,axis=0);fw={h:np.log(px.shift(-h)/px) for h in [5,10,20,40]};rows=[];turn=[];prev=None
for dt in f.index:
 a=f.loc[dt];y=fw[10].loc[dt];ok=a.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(y[ok]),ok.sum()))
  if prev is not None:turn.append((a[ok].rank(pct=True)-prev[ok].rank(pct=True)).abs().mean())
  prev=a
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');x=z.ic
print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15);print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(x.mean(),x.mean()/x.std(),(x>0).mean(),np.nanmean(turn)))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=x[(x.index>=lo)&(x.index<=hi+'-12-31')];print(lo,hi,len(q),q.mean(),q.mean()/q.std(),(q>0).mean())
for h in [5,10,20,40]:
 v=[]
 for dt in f.index:
  a=f.loc[dt];y=fw[h].loc[dt];ok=a.notna()&y.notna()
  if ok.sum()>=8:v.append(a[ok].corr(y[ok]))
 print('decay',h,np.nanmean(v),len(v))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340721_tail_resilient_signal.csv',index=False)
