import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def dat(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);return d[d.date<='2026-07-15'].set_index('date').close.sort_index()
prices={s:dat(s) for s in U}; dates=sorted(set().union(*[set(x.index) for x in prices.values()]))
R=pd.DataFrame({s:x.pct_change() for s,x in prices.items()})
for look in [3,5]:
 Fs={}; Fhs={}
 for s,x in prices.items():
  r=x.pct_change(); v=r.rolling(20,min_periods=15).std(); rr=r.rolling(look,min_periods=look).sum()
  Fs[s]=-rr/v; Fhs[s]=x.shift(-1)/x-1
 F=pd.DataFrame(Fs); FW=pd.DataFrame(Fhs)
 med=F.median(axis=1); F=F.sub(med,axis=0) # relative to factor cross-sectional median (equiv robust relative return after scaling)
 rows=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],FW.loc[dt]],axis=1).dropna();z.columns=['f','y']
  if len(z)>=8 and z.f.nunique()>1: rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');ic=q.ic
 print('look',look,'dates',len(ic),'avgN',q.n.mean(),'coverage',q.n.sum()/(len(ic)*15),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean())
 for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
  x=ic.loc[a:b];print('regime',a,b,'IC',x.mean(),'ICIR',x.mean()/x.std(),'n',len(x))
 for h in [5,10]:
  vals=[]
  for s,x in prices.items(): Fhs[s]=x.shift(-h)/x-1
  fh=pd.DataFrame(Fhs)
  for dt in F.index:
   z=pd.concat([F.loc[dt],fh.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  xx=pd.Series(vals).dropna();print('decay',h,xx.mean(),xx.mean()/xx.std(),len(xx))
