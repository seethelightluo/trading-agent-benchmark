import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5200)
 if d is None or len(d)==0: d=get_index_daily_data(s,5200)
 if d is not None and len(d): P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=np.log(px).diff()
m10=r.rolling(10,min_periods=8).sum(); vol=r.rolling(30,min_periods=20).std()+1e-9
breadth=(m10>0).mean(axis=1)
# Cross-asset breadth-extreme reversal: reverse recent winners/losers only
# when market breadth is decisive; lagged one day and volatility normalized.
gate=np.clip(np.abs(breadth-.5)*2,0,1)
sig=(-m10/vol).mul(gate,axis=0)
sig=sig.sub(sig.mean(axis=1),axis=0).shift(1)
fw={h:np.log(px.shift(-h)/px) for h in [5,10,20,40]}
rows=[]; turns=[]; prev=None
for dt in sig.index:
 a=sig.loc[dt]; y=fw[10].loc[dt]; ok=a.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(y[ok]),ok.sum()))
  if prev is not None:
   q=ok&prev.notna(); turns.append((a[q].rank(pct=True)-prev[q].rank(pct=True)).abs().mean())
  prev=a
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=z.ic.dropna()
print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15)
print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(x.mean(),x.mean()/x.std(),(x>0).mean(),np.nanmean(turns)))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=x[(x.index>=lo)&(x.index<=hi+'-12-31')];print(lo,hi,len(q),q.mean(),q.mean()/q.std() if q.std()>0 else np.nan,(q>0).mean())
for h in [5,10,20,40]:
 vals=[]
 for dt in sig.index:
  a=sig.loc[dt]; y=fw[h].loc[dt]; ok=a.notna()&y.notna()
  if ok.sum()>=8: vals.append(a[ok].corr(y[ok]))
 print('decay',h,np.nanmean(vals),len(vals))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20340526_breadth_extreme_reversal_signal.csv',index=False)
