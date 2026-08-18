import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-04-14'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); v5=r.rolling(5,min_periods=4).std(); v30=r.rolling(30,min_periods=20).std(); ret10=np.log(px/px.shift(10)); f=(ret10*(v30/(v5+1e-12))).shift(1); fr=np.log(px.shift(-10)/px)
ics=[];ns=[];dates=[];turn=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(d)
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turn.append((z.iloc[:,0].rank()-z.iloc[:,1].rank()).abs().mean()/len(z))
s=pd.Series(ics,index=dates).dropna(); print('assets',len(raw),'calendar_dates',len(px),'valid_dates',len(s),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(np.array(ns)/15),4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round(np.mean(s>0),4),'turn',round(np.mean(turn),4))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2032','2033')]:
 q=s[(s.index.year>=int(a))&(s.index.year<=int(b))]; print(a,b,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6) if len(q)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330415_compression_breakout_signal.csv',index=False)
print('signal_path=scripts/miner_3_20330415_compression_breakout_signal.csv')
