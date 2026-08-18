import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-09-20'); b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:end].ffill()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(P.index).ffill()
# Volatility-shock recovery momentum: positive medium-term momentum is favored only
# after an elevated VIX shock that is now easing. All inputs are observable at date t.
vr=v.rolling(252,min_periods=60).rank(pct=True)
easing=(v.pct_change(5)<0).astype(float)
shock=(vr-0.65).clip(lower=0)/0.35
f=P.pct_change(10).mul(1+1.5*shock*easing,axis=0)

def calc(h):
 y=P.shift(-h)/P-1; a=[]; ds=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 a=np.asarray(a); ds=np.asarray(ds,dtype='datetime64[ns]')
 return a,ds,ns
for h in [1,3,5,10,20]:
 a,d,n=calc(h); print('H',h,'dates',len(a),'avgN',np.mean(n),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
a,d,n=calc(10)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-09-20')]:
 q=(d>=np.datetime64(lo))&(d<=np.datetime64(hi));x=a[q];print('REG',lo,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
r=f.rank(axis=1,pct=True); print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',(r-r.shift()).abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_1_20280921_volshock_recovery10_signal.csv')
print('period',P.index.min().date(),P.index.max().date())
