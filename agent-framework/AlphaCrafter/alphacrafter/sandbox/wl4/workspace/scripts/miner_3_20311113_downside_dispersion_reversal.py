import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-11-12'); p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); p[s]=d.close[d.index<=cut]
pd0=pd.DataFrame(p).sort_index(); r=pd0.pct_change(); r5=pd0.pct_change(5); res=r5.sub(r5.median(axis=1),axis=0); disp=r5.std(axis=1); gate=(disp>disp.rolling(60,min_periods=20).median()).astype(float)
down=r.where(r<0,0).rolling(40,min_periods=20).std(); total=r.rolling(40,min_periods=20).std(); f=(-res/(.7*down+.3*total+1e-8)*gate).shift(1); fr=pd0.shift(-10)/pd0-1; v=[]; ds=[]; ns=[]
for t in f.index:
 z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):v.append(q);ds.append(t);ns.append(len(z))
x=pd.Series(v,index=ds);print('factor downside_dispersion_gate_reversal_10d');print('dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 y=x[x.index>=x.index.max()-pd.Timedelta(days=n)];print('recent',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/pd0.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'price_dates',len(pd0),'instruments',len(U))