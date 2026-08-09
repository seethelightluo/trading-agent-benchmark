import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
B='../persistent/stock_data'; R=pd.DataFrame({s:pd.read_csv(f'{B}/{s}.csv',parse_dates=['date']).set_index('date').close.pct_change() for s in U})
d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.pct_change()
b=R.rolling(60,min_periods=45).cov(d).div(d.rolling(60,min_periods=45).var(),axis=0)
F=-b
for w in [20,60,120]:
 X=R.rolling(w,min_periods=max(10,w//2)).corr(d) if False else None
ics=[]; ns=0
for i in range(len(F)-1):
 z=pd.concat([F.iloc[i],R.iloc[i+1]],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns+=1
v=np.array(ics);print('dates',ns,'avg names',len(U),'coverage',ns/len(F),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
for h in [5,10]:
 q=[]
 for i in range(len(F)-h):
  z=pd.concat([F.iloc[i],R.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print(h,q.mean(),q.mean()/q.std(ddof=1),len(q))
for y in range(2020,2027):
 q=[ics[i] for i in range(len(ics)) if F.index[i].year==y]
 if q:print(y,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
