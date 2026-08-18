import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2031-11-26')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d.close[d.index<=cutoff]
px=pd.DataFrame(px).sort_index()
r=px.pct_change()
r10=px.pct_change(10)
resid=r10.sub(r10.median(axis=1),axis=0)
# downside risk over the same recent window; penalize reversals in unstable assets
neg=r.where(r<0,0.0)
down=neg.pow(2).rolling(20,min_periods=12).mean().pow(.5)
cs_down=down.median(axis=1)
# Relative downside risk, bounded to avoid extreme crypto effects
risk=(down.div(cs_down.replace(0,np.nan),axis=0)).clip(.5,2.5)
raw=(-resid).div(risk)
f=raw.rolling(3,min_periods=2).mean().shift(1)
for h in [5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 x=pd.Series(vals)
 print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 if h==10:
  for n in [365,730,1095]:
   y=x.iloc[-n:]; print('recent',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'price_dates',len(px),'instruments',len(U),'cutoff',cutoff.date())
