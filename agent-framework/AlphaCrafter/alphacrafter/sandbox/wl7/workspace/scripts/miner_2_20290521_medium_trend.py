import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 try: d=get_index_daily_data(s,2600)
 except Exception: d=None
 if d is None or len(d)<150:
  try: d=get_stock_daily_data(s,2600)
  except Exception: d=None
 if d is not None: frames[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(frames).sort_index().ffill(); r=p.pct_change()
down=r.where(r<0,0).rolling(60,min_periods=30).std()
f=(p.shift(5)/p.shift(65)-1)/down.shift(5); f=f.replace([np.inf,-np.inf],np.nan)
ics=[]; rows=[]
for i in range(65,len(p)-10):
 a=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; z=pd.concat([a,y],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); rows.append([p.index[i],len(z),ics[-1]])
ics=np.array(ics); dates=pd.to_datetime([x[0] for x in rows]); turn=np.nanmean(np.abs(f.diff()).sum(axis=1)/f.notna().sum(axis=1))
print('candidate medium 60d trend excluding 5d / downside vol60')
print('assets',len(frames),'dates',len(ics),'avg_n',np.mean([x[1] for x in rows]),'coverage',np.mean([x[1] for x in rows])/15)
print('IC',round(np.nanmean(ics),6),'ICIR',round(np.nanmean(ics)/np.nanstd(ics,ddof=1),6),'hit',round(np.mean(ics>0),4),'turnover',round(turn,4))
for label,mask in [('2025-26',(dates>='2025-01-01')&(dates<'2027-01-01')),('2027-28',(dates>='2027-01-01')&(dates<'2029-01-01')),('recent',(dates>='2028-09-01'))]:
 q=ics[mask]; print(label,len(q),'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1),6))
pd.DataFrame(rows,columns=['date','n','ic']).to_csv('scripts/miner_2_20290521_medium_trend_signal.csv',index=False)
