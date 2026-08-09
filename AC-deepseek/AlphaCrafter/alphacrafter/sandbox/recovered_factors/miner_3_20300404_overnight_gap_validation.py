import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2030-04-03'
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().close for a in A}; p=pd.DataFrame(P).loc[:END]
o=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().open for a in A}).loc[p.index]
r=p.pct_change(); sig=(-(o/p.shift(1)-1).rolling(3,min_periods=2).mean()).shift(1)
for h in [1,5,10,20]:
 q=[]; ns=[]
 for i in range(len(p)-h):
  z=pd.concat([sig.iloc[i],(p.iloc[i+h]/p.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8 and z.nunique().min()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(q);print('H',h,'IC %.6f ICIR %.6f dates %d hit %.4f meanN %.2f'%(x.mean(),x.mean()/x.std(ddof=1),len(x),np.mean(x>0),np.mean(ns)))
vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/vol; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); kurt=-r.rolling(40,min_periods=30).kurt(); es=-r.rolling(40,min_periods=30).quantile(.2)/vol
mx=0;who=''
for n,s in {'trend':trend,'reversal':rev,'kurtosis':kurt,'es':es}.items():
 z=pd.concat([sig.stack().rename('x'),s.stack().rename('y')],axis=1).dropna(); rho=z.x.corr(z.y,method='spearman'); print('rho',n,rho,len(z));
 if abs(rho)>mx:mx=abs(rho);who=n
print('candidate_max_abs_library_correlation',mx,who)
