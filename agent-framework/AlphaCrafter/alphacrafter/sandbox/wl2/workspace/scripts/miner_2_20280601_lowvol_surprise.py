import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d['date']=pd.to_datetime(d.date); p[a]=d.sort_values('date').set_index('date').close.astype(float)
pd_=pd.DataFrame(p).sort_index(); r=pd_.pct_change()
# low-volatility with volatility surprise: favor assets whose recent risk is below their own medium baseline
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
f=(-v20/(v60+1e-12)).shift(1)
for h in [1,3,5,10]:
 y=pd_.pct_change(h).shift(-h); out=[]
 for dt in pd_.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(out,columns=['d','n','ic']).set_index('d'); ic=q.ic
 print('h',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 for name,s in [('early',q.iloc[:int(len(q)*.6)]),('recent',q.iloc[int(len(q)*.6):])]:
  x=s.ic; print(name,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),len(x))
print('coverage',round(f.notna().sum().sum()/(len(pd_)*len(A)),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
