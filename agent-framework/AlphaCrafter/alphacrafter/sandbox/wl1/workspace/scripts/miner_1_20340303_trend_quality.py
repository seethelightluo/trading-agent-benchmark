import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
down=r.clip(upper=0).rolling(40,min_periods=20).std()*np.sqrt(40)
rv=r.rolling(40,min_periods=20).std()*np.sqrt(40)
# interpretable trend quality: medium return penalized by downside risk, with recent reversal overlay
base=np.log(px/px.shift(60))/(down+1e-8)
short=np.log(px/px.shift(10))
for a in [0,.25,.5,.75,1.0]:
 f=(base-a*short/(rv+1e-8)).shift(1)
 for h in [5,10,20]:
  fr=px.pct_change(h).shift(-h); z=[]; cov=[]
  for dt in f.index:
   q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));cov.append(len(q)/15)
  z=np.array(z)
  print('a',a,'h',h,'dates',len(z),'avgN',round(np.mean(cov)*15,2),'IC %.8f ICIR %.8f hit %.4f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0)))
  if a==.5 and h==20:
   out=f.copy();out.insert(0,'date',out.index);out.to_csv('scripts/miner_1_20340303_trend_quality_signal.csv',index=False)
