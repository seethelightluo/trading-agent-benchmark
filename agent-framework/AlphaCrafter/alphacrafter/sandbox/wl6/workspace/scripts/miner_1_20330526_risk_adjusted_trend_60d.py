import pandas as pd, numpy as np, glob
from pathlib import Path
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
    D[a]=x['close'].pct_change()
R=pd.DataFrame(D).sort_index()
# only information through t: medium trend normalized by recent realized risk
mom=R.rolling(60,min_periods=45).sum()
vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
f=mom/vol.replace(0,np.nan)
# forward 10 trading-day return
fr=R.shift(-1).rolling(10,min_periods=10).sum().shift(-9)
ics=[]; ns=[]; turnovers=[]
prev=None
for dt in f.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8:
        ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
    sig=f.loc[dt].rank(pct=True)
    if prev is not None:
        turnovers.append((sig-prev).abs().mean())
    prev=sig
s=pd.Series(ics).dropna()
print('dates',len(s),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turnover',np.nanmean(turnovers))
for h in [5,10,20,40]:
    fh=R.shift(-1).rolling(h,min_periods=h).sum().shift(-(h-1)); q=[]
    for dt in f.index:
      z=pd.concat([f.loc[dt],fh.loc[dt]],axis=1).dropna()
      if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    q=pd.Series(q).dropna(); print('decay',h,q.mean(),q.mean()/q.std())
