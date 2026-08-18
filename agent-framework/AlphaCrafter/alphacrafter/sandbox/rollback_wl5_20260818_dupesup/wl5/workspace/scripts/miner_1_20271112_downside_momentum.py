import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  q=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:pd.Timestamp('2027-11-12')]; D[s]=q
 except Exception as e: print('missing',s)
# downside-risk-adjusted medium momentum, with cross-sectional sign expected positive
rows=[]; series={}
for s,x in D.items():
 r=x.close.pct_change(); down=r.clip(upper=0).rolling(30).std()
 series[s]=x.close.pct_change(20)/(down*np.sqrt(20)+1e-9)
px=pd.DataFrame({s:x.close for s,x in D.items()}); fac=pd.DataFrame(series); fwd=px.shift(-10)/px-1
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=np.array([v for d,v,n in rows]); dates=[d for d,v,n in rows]
print('candidate downside_momentum_20d_down30 dates',len(a),'avgN',np.mean([n for d,v,n in rows]),'coverage',np.mean([n for d,v,n in rows])/15)
print('IC',a.mean(),'std',a.std(ddof=1),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-11-12')]:
 q=[v for d,v,n in rows if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)]; print('regime',lo,len(q),np.mean(q) if q else None)
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; aa=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,len(aa),np.mean(aa))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('assets',len(D))
