import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-12-10'); p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index();p[s]=d.close[d.index<=cut]
px=pd.DataFrame(p).sort_index();r=px.pct_change();ret=px.pct_change(20);res=ret.sub(ret.median(axis=1),axis=0);v20=r.rolling(20,min_periods=15).std();v60=r.rolling(60,min_periods=40).std();rank=(v20/(v60+1e-12)).rank(axis=1,pct=True);f=(-res/(v60+1e-12)*(1-rank).clip(.1,.9)).shift(1)
for h in [5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for t in f.index:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):vals.append(q);ns.append(len(z))
 x=pd.Series(vals);print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 fr=px.shift(-10)/px-1;vals=[]
 for t in f.index[-n:]:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals);print('recent',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'price_dates',len(px),'instruments',len(U),'cutoff',cut.date())