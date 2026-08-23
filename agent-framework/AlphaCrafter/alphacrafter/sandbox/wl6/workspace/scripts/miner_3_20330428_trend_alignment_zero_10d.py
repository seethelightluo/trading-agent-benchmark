import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in A}
p=pd.concat(px,axis=1).sort_index().loc[:'2033-04-27']; r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
r20=p.pct_change(20); r60=p.pct_change(60)
sig=(r20/vol)*np.where(np.sign(r20)==np.sign(r60),1.0,0.0)
print('candidate trend_alignment_zero; instruments',len(A),'last',p.index[-1].date())
for h in [5,10,20,40]:
 f=p.shift(-h).div(p)-1; vals=[];ns=[];dates=[]
 for i in range(len(p)-h):
  z=pd.concat([sig.iloc[i],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z));dates.append(p.index[i])
 x=np.asarray(vals); print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.asarray(ns)/15),4))
ser=sig.rank(axis=1,pct=True); print('turnover',round(float(np.nanmean(np.abs(ser.diff()).mean(axis=1))),6))
