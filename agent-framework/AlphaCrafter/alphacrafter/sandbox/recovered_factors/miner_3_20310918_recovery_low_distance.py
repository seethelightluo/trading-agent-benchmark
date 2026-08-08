import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
D={os.path.basename(f)[:-4]:pd.read_csv(f).set_index('date')['close'] for f in files}
prices=pd.DataFrame(D).sort_index(); rets=prices.pct_change()
# Recovery-from-recent-low: distance above 20-day low, normalized by 20d vol.
low=prices.rolling(20,min_periods=15).min(); vol=rets.rolling(20,min_periods=15).std()
f=(prices/low-1)/(vol*np.sqrt(20)+1e-12)
print('assets',len(prices.columns),'dates',len(prices),'range',prices.index.min(),prices.index.max())
for h in [1,5,10,20]:
  vals=[]; ns=[]
  for i in range(len(prices)-h):
    x=f.iloc[i]; y=prices.iloc[i+h]/prices.iloc[i]-1
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
      vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  a=np.array(vals); print('H',h,'dates',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
  for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
   q=[v for d,v in zip(prices.index[:-h],vals) if lo<=d[:4]<=hi] # approximate mismatch due skips
  recent=a[-120:]; print(' recent',recent.mean(),recent.mean()/recent.std(ddof=1))
# turnover 10 day rank changes
rank=f.rank(axis=1,pct=True); print('turn10',((rank-rank.shift(10)).abs().mean(axis=1)).mean())
print('coverage',f.notna().mean().mean())
