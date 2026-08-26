import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2034-12-06']; r=np.log(P).diff()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(P.index).ffill()
# relief: contrarian short-term reversal is strongest after elevated VIX and falling VIX
vp=v.rolling(252,min_periods=60).rank(pct=True); relief=(v.pct_change(2)<0)&(vp.shift(1)>0.70)
base=-r.rolling(5).sum()/r.rolling(20).std()
sig=base.mul((1+0.75*relief.astype(float)),axis=0).shift(1)
fwd=P.shift(-10)/P-1
ics=[];ns=[];ds=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: ics.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()));ns.append(len(z));ds.append(dt)
ic=pd.Series(ics,index=pd.to_datetime(ds)).dropna()
print({'dates':len(ic),'avg_N':float(np.mean(ns)),'coverage':float(np.mean(ns)/15),'ic_10d':float(ic.mean()),'daily_icir':float(ic.mean()/ic.std(ddof=1)),'hit':float((ic>0).mean()),'turnover':float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())})
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; aa=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: aa.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('decay',h,float(np.nanmean(aa)),len(aa))
print('recent365_icir',float(ic.tail(365).mean()/ic.tail(365).std(ddof=1)))
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20341207_vix_relief_reversal_signal.csv',index=False)
ic.rename('ic').to_csv('scripts/miner_2_20341207_vix_relief_reversal_ic.csv')
