import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-11-02'); frames={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]; d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); frames[s]=d.close
px=pd.DataFrame(frames).sort_index(); pxsig=px.loc[:cut]; ret=pxsig.pct_change(); r5=pxsig.pct_change(5)
vol=ret.rolling(20,min_periods=15).std()*np.sqrt(252)
raw=r5.sub(r5.median(axis=1),axis=0); factor=-raw/vol
fwd=px.shift(-10)/px-1; rows=[]
for dt in factor.index:
 if dt>cut: continue
 z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=np.array([x[1] for x in rows]); ns=np.array([x[2] for x in rows])
print('cut',cut.date(),'dates',len(a),'avgN',ns.mean(),'coverage',ns.mean()/15)
print('IC',a.mean(),'std',a.std(ddof=1),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-11-02')]:
 q=[v for d,v,n in rows if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)]; print('regime',lo,hi,'n',len(q),'ic',np.mean(q) if q else None,'icir',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else None)
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; aa=[]
 for dt in factor.index:
  if dt>cut: continue
  z=pd.concat([factor.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'IC',np.mean(aa),'n',len(aa))
r=factor.rank(axis=1,pct=True); print('turnover_rank_proxy',r.diff().abs().mean(axis=1).loc[:cut].mean())
mid=len(a)//2; print('halves',np.mean(a[:mid]),np.mean(a[mid:]),len(a[:mid]),len(a[mid:]))
