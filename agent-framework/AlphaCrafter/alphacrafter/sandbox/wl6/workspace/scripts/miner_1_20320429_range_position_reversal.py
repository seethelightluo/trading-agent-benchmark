import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); D[a]=x
p=pd.DataFrame({a:D[a].close for a in A}).sort_index(); hi=pd.DataFrame({a:D[a].high for a in A}).reindex(p.index); lo=pd.DataFrame({a:D[a].low for a in A}).reindex(p.index)
# 10-day close location within prior 40-day range, lagged one day; low location is reversal candidate
rh=hi.shift(1).rolling(40,min_periods=25).max(); rl=lo.shift(1).rolling(40,min_periods=25).min(); f=(-(p.shift(1)-rl)/(rh-rl)).clip(-1,0)
# map to numeric: extreme low gets high score, avoid zero-range
f=(-(p.shift(1)-rl)/(rh-rl).replace(0,np.nan)).clip(-1,1)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; I=[];N=[];T=[]; prev=None
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:I.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);N.append(len(z))
  rk=f.loc[dt].rank(pct=True)
  if prev is not None:
   q=pd.concat([rk,prev],axis=1).dropna();T.append(np.mean(abs(q.iloc[:,0]-q.iloc[:,1])))
  prev=rk
 I=np.array(I);print({'horizon':h,'dates':len(I),'avg_n':round(np.mean(N),2),'coverage':round(np.mean(N)/15,4),'ic':round(np.mean(I),6),'icir':round(np.mean(I)/np.std(I,ddof=1),4),'hit':round(np.mean(I>0),4),'turnover':round(np.nanmean(T),4)})
print('data_end',p.index.max().date())
