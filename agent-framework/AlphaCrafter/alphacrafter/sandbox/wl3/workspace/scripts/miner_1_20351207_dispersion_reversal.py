import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-12-06'); C={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);C[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index().loc[:cutoff].ffill();R=P.pct_change()
# short reversal strengthened when cross-sectional dispersion is high; all inputs lagged
csdisp=R.rolling(20,min_periods=15).std().mean(axis=1)
F=(-R.rolling(3,min_periods=3).sum()).mul((csdisp/csdisp.rolling(60,min_periods=30).median()).clip(.5,2),axis=0).shift(1)
for h in [1,5,10,20]:
 vals=[];dates=[]; counts=[]
 for i in range(65,len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna();counts.append(len(z))
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):vals.append(q);dates.append(P.index[i])
 a=pd.Series(vals,index=dates);print('h',h,'dates',len(a),'avgN',round(np.mean(counts),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'recent',round(a.tail(252).mean(),6),round(a.tail(252).mean()/a.tail(252).std(ddof=1),6))
print('coverage',round(F.notna().mean().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
