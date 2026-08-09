import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-08-27')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:cut]; px[s]=d.close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); short=R.rolling(5,min_periods=4).std(); long=R.rolling(60,min_periods=40).std(); F=-(short/long)
rows=[]
for dt in F.index:
 for h in [1,5,10]:
  z=pd.concat([F.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','h','n','ic']); print('dates/n/cov',df[df.h==1].date.nunique(),round(df[df.h==1].n.mean(),2),round(df[df.h==1].n.mean()/15,4))
for h in [1,5,10]:
 q=df[df.h==h].ic; print('H',h,'obs',len(q),'IC',round(q.mean(),5),'ICIR_raw',round(q.mean()/q.std(ddof=1),5),'ICIR_ann',round(q.mean()/q.std(ddof=1)*np.sqrt(252),4),'hit',round((q>0).mean(),4))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026-08-27')]:
 q=df[(df.h==1)&(df.date>=a)&(df.date<=b)].ic; print('regime',a,b,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
