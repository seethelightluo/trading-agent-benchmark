import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2031-10-15'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d.close[d.index<=cutoff]
px=pd.DataFrame(px).sort_index(); r=px.pct_change()
r20=px.pct_change(20); resid=r20.sub(r20.median(axis=1),axis=0)
down=r.where(r<0,0).rolling(40,min_periods=25).std(); total=r.rolling(40,min_periods=25).std()
risk=(0.7*down+0.3*total).replace(0,np.nan)
base=-resid/(risk+1e-8)
breadth=(r20>0).mean(axis=1).rolling(30,min_periods=20).mean()
mult=(1.35-1.0*breadth).clip(.65,1.35)
f=base.mul(mult,axis=0).shift(1)
fr={h:px.shift(-h)/px-1 for h in [5,10,20]}
for h in [5,10,20]:
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(z))
 x=pd.Series(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 vals=[]
 for dt in f.index[-n:]:
  z=pd.concat([f.loc[dt],fr[10].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('recent',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'price_dates',len(px),'instruments',len(U),'cutoff',cutoff.date())
