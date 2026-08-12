import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);raw[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(raw).sort_index().ffill(limit=3); r=np.log(p).diff()
# medium trend quality: 60d return, penalized by 20d volatility and choppiness
mom=np.log(p/p.shift(60)); vol=r.rolling(20,min_periods=15).std(); eff=mom/(vol*np.sqrt(60)); f=(eff - r.rolling(60,min_periods=30).std()).shift(1); f=f.sub(f.mean(axis=1),axis=0)
for h in [1,3,5,10]:
 fr=np.log(p.shift(-h)/p); z=[];ds=[];ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8:z.append(f.loc[dt,ok].corr(fr.loc[dt,ok],method='spearman'));ds.append(dt);ns.append(ok.sum())
 z=pd.Series(z,index=ds).dropna(); print('H',h,'dates',len(z),'avg_n',round(np.mean(ns),2),'IC',round(z.mean(),7),'ICIR',round(z.mean()/z.std(ddof=1),7),'hit',round((z>0).mean(),4))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4));f.to_csv('scripts/miner_1_20290823_trend_quality_signal.csv')
