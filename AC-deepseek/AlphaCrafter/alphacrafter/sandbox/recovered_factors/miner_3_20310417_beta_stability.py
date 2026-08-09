import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
R={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 R[a]=np.log(d.close).diff()
r=pd.DataFrame(R); m=r.median(axis=1)
# instability of rolling 20d beta to cross-asset median, 60d window; higher score = stable beta
cov=r.rolling(20,min_periods=15).cov(m); var=m.rolling(20,min_periods=15).var()
beta=cov.div(var,axis=0)
f=1/(beta.rolling(60,min_periods=40).std()+0.05)
# neutralize daily cross-section against beta level to isolate stability
f=f.sub(f.mean(axis=1),axis=0)
F={a:f[a] for a in A}; df=pd.concat(F,axis=1)
for h in [1,5,10,20]:
 vals=[]; ns=[]
 fr=r.shift(-h).rolling(h).sum().shift(-(h-1))
 for dt in df.index:
  z=pd.concat([df.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 v=np.array(vals); print('H',h,'dates',len(v),'meanN',np.mean(ns),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
for y0,y1 in [('2020','2023'),('2024','2027'),('2028','2030'),('2030','2031')]:
 vals=[]
 fr=r.shift(-1)
 for dt in df.index:
  if y0<=dt.strftime('%Y')<=y1:
   z=pd.concat([df.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 v=np.array(vals);print('REG',y0,y1,len(v),v.mean(),v.mean()/v.std(ddof=1) if len(v)>1 else np.nan)
print('coverage',df.notna().mean().mean(),'dates',len(df),'meanN',df.notna().sum(axis=1).mean())
