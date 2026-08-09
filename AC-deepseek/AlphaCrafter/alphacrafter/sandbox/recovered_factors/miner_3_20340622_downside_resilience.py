import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for a in assets:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close']
 d[a]=x
px=pd.DataFrame(d).sort_index().loc[:'2034-06-21']
r=np.log(px/px.shift(1)); market=r.mean(axis=1)
# Defensive downside resilience: mean asset return on prior 60 sessions when cross-asset market was down,
# with shrinkage toward unconditional mean to avoid sparse conditional samples.
down=market<0
n=down.rolling(60).sum()
f=(r.where(down).rolling(60,min_periods=15).mean()*n/(n+10) + r.rolling(60,min_periods=30).mean()*10/(n+10))
f=f.shift(1)
# cross-sectional IC, forward horizons
for h in [1,5,10,20]:
 fr=np.log(px.shift(-h)/px)
 vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(dt)
 s=pd.Series(vals,index=dates)
 print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(s.mean(),s.mean()/s.std(),(s>0).mean(),len(s),np.mean(ns)))
 for p,q in [('2020-2023','2023-12-31'),('2024-2027','2027-12-31'),('2028-2030','2030-12-31'),('2031-2034','2034-06-21')]:
  ss=s.loc[p:q]; print(p, '%.6f'%ss.mean(),len(ss))
# turnover proxy: rank changes / 10 sessions
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(10)).abs().mean(axis=1).mean(); print('coverage',f.notna().mean().mean(),'mean valid',f.notna().sum(axis=1).mean(),'turn10',turn)
print('last',f.tail(1).T.to_string())
