import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 x=pd.read_csv(f); x.date=pd.to_datetime(x.date); x=x[x.date<='2026-07-15'];x=x.set_index('date').sort_index()
 # candle pressure: close location within daily range, range scaled by recent ATR to avoid tiny candles
 rng=(x.high-x.low).replace(0,np.nan)
 clv=(2*x.close-x.high-x.low)/rng
 relrng=rng/x.close / (rng/x.close).rolling(20,min_periods=10).median()
 d[s]=pd.DataFrame({'f':(clv*relrng).rolling(5,min_periods=5).mean(),'r':x.close.pct_change()})
# aligned panel and forward 1d
F=pd.concat({s:v.f for s,v in d.items()},axis=1); R=pd.concat({s:v.r.shift(-1) for s,v in d.items()},axis=1)
ics=[]; dates=[]
for dt in F.index:
 a=F.loc[dt]; b=R.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt)
ic=np.array(ics); print('dates',len(ic),'mean_n',np.mean([sum(F.loc[x].notna()&R.loc[x].notna()) for x in dates]),'IC',np.nanmean(ic),'ICIR',np.nanmean(ic)/np.nanstd(ic,ddof=1),'hit',np.mean(ic>0),'coverage',F.notna().sum(axis=1).mean()/15)
for h in [1,5,10]:
 RR=pd.concat({s:d[s].r.rolling(h).sum().shift(-h) for s in U},axis=1); zics=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],RR.loc[dt]],axis=1).dropna()
  if len(z)>=8:zics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 zics=np.array(zics);print('h',h,'N',len(zics),'IC',np.nanmean(zics),'ICIR',np.nanmean(zics)/np.nanstd(zics,ddof=1),'hit',np.mean(zics>0))
# turnover ranks
rank=F.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
# regimes yearly
for y in sorted(set(x.year for x in dates)):
 q=ic[np.array([x.year==y for x in dates])];print(y,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1),4) if len(q)>1 else None)
# corr with existing factors approximate values by loading json expressions unavailable; report calc corr against factor snapshots if scripts no
