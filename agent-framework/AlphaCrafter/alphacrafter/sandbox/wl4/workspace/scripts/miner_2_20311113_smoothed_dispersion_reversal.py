import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-11-12')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close[d.index<=cutoff]
px=pd.DataFrame(px).sort_index(); r10=px.pct_change(10); resid=r10.sub(r10.median(axis=1),axis=0)
vol=r10.std(axis=1); baseline=vol.rolling(40,min_periods=25).mean(); gate=(vol/(baseline+1e-8)).clip(.5,2.0)
raw=-resid.mul(gate,axis=0)
# 3-session trailing average, then lag one complete session
f=raw.rolling(3,min_periods=2).mean().shift(1)
frs={h:px.shift(-h)/px-1 for h in [5,10,20]}
for h,fr in frs.items():
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 x=pd.Series(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 if h==10:
  for n in [365,730,1095]:
   y=x.iloc[-n:];print('recent',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'price_dates',len(px),'instruments',len(U),'cutoff',cutoff.date())
