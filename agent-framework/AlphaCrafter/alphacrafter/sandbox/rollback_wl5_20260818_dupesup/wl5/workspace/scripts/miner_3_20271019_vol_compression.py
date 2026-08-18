import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-10-19'); frames={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]; d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
 if 'close' in d: frames[s]=d.close
px=pd.DataFrame(frames).loc[:cut]
# volatility compression: inverse 20d realized vol, with 60d cross-sectional normalization implicit by ranks
vol=px.pct_change().rolling(20).std(); factor=-vol
fwd=px.shift(-10)/px-1; rows=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=np.array([x[1] for x in rows]); print('dates',len(a),'avgN',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15); print('IC',a.mean(),'std',a.std(ddof=1),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-10-19')]:
 q=[v for d,v,n in rows if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)]; print(lo,len(q),np.mean(q) if q else None)
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; aa=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(aa),len(aa))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
