import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 if d is None or len(d)==0:return None
 d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index().close.rename(s)
xs=[get(s) for s in U]; p=pd.concat([x for x in xs if x is not None],axis=1).sort_index(); lr=np.log(p).diff()
# Smoothed volatility-normalized trend acceleration: average of recent 5 daily observations,
# using only information through date t. Cross-sectional Spearman IC versus forward returns.
raw=(p.pct_change(20)-p.pct_change(60))/(lr.rolling(60,min_periods=40).std()*np.sqrt(60))
f=raw.rolling(5,min_periods=5).mean()
print('instruments',p.shape[1],'price_dates',len(p),'range',p.index.min(),p.index.max())
for h in [5,10,20]:
 vals=[]
 for i in range(65,len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append((p.index[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print('HORIZON',h,'dates',len(a),'avg_n',a.n.mean(),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
 for lab,m in [('2020-22',a.index<'2023'),('2023-25',(a.index>='2023')&(a.index<'2026')),('2026-27',(a.index>='2026')&(a.index<'2028')),('2028+',a.index>='2028'),('recent250',np.arange(len(a))>=len(a)-250)]:
  q=a.loc[m,'ic']; print(lab,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('coverage',f.notna().mean().mean(),'turnover',((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift()).abs().mean(axis=1)).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20290531_smoothed_trend_acceleration_signal.csv',index=False)
