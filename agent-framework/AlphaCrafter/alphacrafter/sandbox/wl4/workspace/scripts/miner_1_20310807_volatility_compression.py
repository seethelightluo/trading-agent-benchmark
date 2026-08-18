import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2031-08-07'); p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']; p[s]=d[d.index<=cutoff]
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
for a,b in [(10,60),(20,120),(40,120)]:
 f=-(r.rolling(a).std()/(r.rolling(b).std()+1e-8)).shift(1); print('FACTOR',a,b)
 for h in [5,10,20]:
  fr=p.shift(-h)/p-1; vals=[]; ns=[]
  for dt in f.index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  x=pd.Series(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
  if h==10:
   for n in [365,730,1095]:
    y=x.iloc[-n:]; print('recent',n,'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'dates',len(y))
 print('coverage',round(f.notna().sum().sum()/p.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
