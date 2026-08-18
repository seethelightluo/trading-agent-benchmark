import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 if d is not None: frames[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(frames).sort_index().ffill(); r=p.pct_change(); rv20=r.rolling(20).std(); rv60=r.rolling(60).std()
market=r.mean(axis=1); breadth=(r<0).mean(axis=1); shock=(rv20.mean(axis=1)/rv60.mean(axis=1)-1).shift(1)
res5=p.pct_change(5).sub(p.pct_change(5).mean(axis=1),axis=0).div(rv20.shift(1),axis=0)
g=((breadth.shift(1)>0.60)&(market.rolling(20).mean().shift(1)<0)&(shock>0.15)).astype(float)
f=-res5.mul(g,axis=0).replace([np.inf,-np.inf],np.nan); fr=p.pct_change().shift(-1)
ics=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt)
ics=np.array(ics); active=g>0
print('assets',len(frames),'dates',len(p),'IC dates',len(ics),'active',int(active.sum()),'coverage',float(np.mean([f.loc[d].notna().mean() for d in f.index if active.loc[d]])))
print('IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(ics>0))
for h in [3,5,10]:
 yy=p.pct_change(h).shift(-h); aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(h,'d',len(aa),'IC',np.nanmean(aa),'ICIR',np.nanmean(aa)/np.nanstd(aa,ddof=1))
print('earlylate',np.nanmean(ics[:len(ics)//2]),np.nanmean(ics[len(ics)//2:]),'period',dates[0],dates[-1])
out=f.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20330401_vol_transition_reversal_signal.csv')
