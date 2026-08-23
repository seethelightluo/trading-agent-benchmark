import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); px[s]=x.close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-04-14']; r=p.pct_change()
# Positive medium-term momentum penalized only by downside volatility; floor avoids unstable ratios.
down=r.where(r<0,0).rolling(40,min_periods=30).std()
f=(p/p.shift(20)-1)/(0.005+down*np.sqrt(20)); f=f.clip(-20,20)
print('cutoff',p.index.max().date(),'dates',len(p),'assets',len(U))
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]; dates=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if pd.notna(q): vals.append(q); ns.append(len(z)); dates.append(dt)
 q=pd.Series(vals,index=dates); print('h',h,'valid',len(q),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,5),'IC',round(q.mean(),8),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6),'hit',round((q>0).mean(),4))
 print('regimes',q.groupby(q.index.year).agg(['mean','count']).round(5).to_dict('index'))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'factor_coverage',round(f.notna().sum(axis=1).mean()/15,5))
