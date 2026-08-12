import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None:return None
 d.date=pd.to_datetime(d.date);return d.set_index('date').sort_index().close.rename(s)
p=pd.concat([get(s) for s in U if get(s) is not None],axis=1).sort_index(); lr=np.log(p).diff()
# trend acceleration, plus cross-sectional rank; test 10-day horizon
f=(p.pct_change(20)-p.pct_change(60))/(lr.rolling(60,min_periods=40).std()*np.sqrt(60))
for h in [5,10,20]:
 a=[]
 for i in range(60,len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append((p.index[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 a=pd.Series(dict(a));print('h',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 for lab,m in [('20-22',a.index<'2023'),('23-25',(a.index>='2023')&(a.index<'2026')),('26-27',(a.index>='2026')&(a.index<'2028')),('28+',a.index>='2028'),('recent250',np.arange(len(a))>=len(a)-250)]:
  q=a[m];print(lab,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20290517_trend_acceleration_signal.csv',index=False)
print('coverage',f.notna().mean().mean(),'turnover',(np.sign(f)!=np.sign(f).shift()).mean(axis=1).mean())
