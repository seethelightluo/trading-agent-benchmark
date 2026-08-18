import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-06-10')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=np.log(p).diff(); lag=r.shift(1)
# defensive low downside-volatility: inverse of lagged 30d downside deviation, with return sign penalty for negative trend
neg=lag.where(lag<0,0).pow(2).rolling(30,min_periods=15).mean().shift(1).pow(.5)
fac=(-neg).replace([np.inf,-np.inf],np.nan)
for h in [1,3,5,10]:
 fr=np.log(p).shift(-h)-np.log(p); vals=[]; ns=[]
 for dt in fac.index:
  a=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); ns.append(len(a))
 x=np.asarray(vals); print(f'h={h} dates={len(x)} avgN={np.mean(ns):.3f} coverage={np.mean(ns)/15:.3f} IC={np.nanmean(x):.8f} ICIR={np.nanmean(x)/np.nanstd(x,ddof=1):.8f} hit={np.mean(x>0):.4f}')
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20330610_downside_vol_signal.csv',index=False)
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
