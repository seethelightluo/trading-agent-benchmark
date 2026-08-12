import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def dat(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);d=d[d.date<='2026-07-15'].set_index('date');return d.close
R=pd.DataFrame({s:dat(s).pct_change() for s in U}).sort_index()
for look in [3,5,10]:
 x=R.rolling(look,min_periods=look).sum(); f=-x.sub(x.median(axis=1),axis=0)
 rows=[]
 for dt in R.index:
  z=pd.concat([f.loc[dt],R.shift(-1).loc[dt]],axis=1).dropna();z.columns=['f','y']
  if len(z)>=8 and z.f.nunique()>1:rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');o=q.ic
 print('look',look,'dates',len(o),'avgN',round(q.n.mean(),2),'IC',round(o.mean(),6),'ICIR',round(o.mean()/o.std(),6),'hit',round((o>0).mean(),4))
 for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
  z=o.loc[a:b];print(a,round(z.mean(),5),round(z.mean()/z.std(),5),len(z))
 for h in [5,10]:
  yy=sum(R.shift(-k) for k in range(1,h+1));v=[]
  for dt in R.index:
   z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  z=pd.Series(v).dropna();print('decay',h,round(z.mean(),6),round(z.mean()/z.std(),6),len(z))
