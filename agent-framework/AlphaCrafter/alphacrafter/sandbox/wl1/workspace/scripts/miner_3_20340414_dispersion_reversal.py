import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,5200)
 if d is None or len(d)==0:d=get_index_daily_data(s,5200)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=np.log(px).diff(); disp=r.std(axis=1).rolling(20,min_periods=15).mean(); z=(disp-disp.rolling(120,min_periods=60).mean())/(disp.rolling(120,min_periods=60).std()+1e-9)
# reversal strengthened in high cross-asset dispersion, risk scaled and lagged
f=(-r.rolling(5,min_periods=5).sum()/(r.rolling(20,min_periods=15).std()+1e-9))*(1+0.7*(z>0).astype(float)); f=f.shift(1); fw=np.log(px.shift(-10)/px)
rows=[];turn=[]
for dt in f.index:
 a=f.loc[dt];y=fw.loc[dt];ok=a.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(y[ok]),ok.sum()))
  if len(rows)>1:
   p=f.loc[rows[-2][0]];q=ok&p.notna();turn.append((a[q].rank(pct=True)-p[q].rank(pct=True)).abs().mean())
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');ic=z.ic.dropna();print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15);print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean(),np.nanmean(turn)))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=ic[(ic.index>=lo)&(ic.index<=hi+'-12-31')];print(lo,hi,len(q),q.mean(),q.mean()/q.std() if q.std()>0 else np.nan,(q>0).mean())
for h in [5,10,20]:
 y=np.log(px.shift(-h)/px);a=[]
 for dt in f.index:
  x=f.loc[dt];v=y.loc[dt];ok=x.notna()&v.notna()
  if ok.sum()>=8:a.append(x[ok].corr(v[ok]))
 print('decay',h,np.nanmean(a),len(a))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20340414_dispersion_reversal_signal.csv',index=False)
