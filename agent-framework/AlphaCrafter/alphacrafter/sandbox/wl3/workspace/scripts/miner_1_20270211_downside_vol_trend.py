import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];a=[]
for s in U:
 d=get_stock_daily_data(s,2400)
 if d is None or len(d)<100:d=get_index_daily_data(s,2400)
 if d is not None:a.append(d[['date','close']].assign(symbol=s))
w=pd.concat(a).pivot(index='date',columns='symbol',values='close').sort_index();r=w.pct_change(); neg=r.clip(upper=0)
down=np.sqrt((neg**2).rolling(20,min_periods=10).mean()); vol=r.rolling(20,min_periods=10).std(); trend=w.pct_change(20)
f=-down/(vol+1e-12)+.25*trend/(vol*np.sqrt(20)+1e-12)
def run(mask):
 qs=[];ns=[]
 for dt in w.index[mask]:
  z=pd.concat([f.loc[dt],(w.shift(-1)/w-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(x):qs.append(x);ns.append(len(z))
 q=pd.Series(qs);return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1), (q>0).mean(),np.mean(ns)
print('cutoff',w.index.max().date(),'dates',len(w),'assets',len(w.columns));print('ALL',run(np.ones(len(w),bool)))
for a1,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-02-10')]:print('REG',a1,b,run((w.index>=a1)&(w.index<=b)))
print('coverage',f.notna().mean().mean(),'turnover',((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift()).abs().mean(axis=1)).mean())
f.stack().rename('signal').reset_index().to_csv('scripts/miner_1_20270211_downside_vol_trend_signal.csv',index=False)
