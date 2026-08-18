import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-12-10'); p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); p[s]=d.close[d.index<=cut]
px=pd.DataFrame(p).sort_index(); r=px.pct_change(); r10=px.pct_change(10); res=r10.sub(r10.median(axis=1),axis=0)
neg=(-r).where(r<0); tail=neg.rolling(40,min_periods=20).mean(); f=(-res/(tail+1e-8)).shift(1)
fr=px.shift(-10)/px-1; vals=[]; dates=[]; ns=[]
for t in f.index:
 z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): vals.append(q);dates.append(t);ns.append(len(z))
x=pd.Series(vals,index=dates);print('factor tailrisk_reversal_10d');print('dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 y=x[x.index>=x.index.max()-pd.Timedelta(days=n)];print('recent',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'price_dates',len(px),'instruments',len(U),'cut',cut.date())