import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-08-27')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index()['close'].astype(float); px[s]=d[d.index<=cut]
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(axis=1)
# residual momentum: 20d cumulative return less 60d rolling beta to equal-weight market times market 20d return
cov=R.rolling(60,min_periods=40).cov(m); var=m.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
f=(P/P.shift(20)-1)-beta*(m.rolling(20).sum())
rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','h','n','ic']); q=df[df.h==1].ic
print('dates/n/cov',q.size,df[df.h==1].n.mean(),df[df.h==1].n.mean()/15)
for h in [1,5,10]:
 q=df[df.h==h].ic; print('H',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026-08-27')]:
 q=df[(df.h==1)&(df.date>=a)&(df.date<=b)].ic;print('regime',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
