import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-09-10')
P=pd.DataFrame({s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index(); P=P.loc[:cut]; R=P.pct_change()
rows=[]
for w in [10,20,40,60]:
 f=-R.rolling(w,min_periods=max(8,int(w*.75))).std()
 for dt in f.index:
  for h in [1,5,10]:
   z=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
   if len(z)>=8: rows.append((w,dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['w','date','h','n','ic'])
for w in [10,20,40,60]:
 q=df[(df.w==w)&(df.h==1)].ic; print('W',w,'obs',len(q),'avgN',df[(df.w==w)&(df.h==1)].n.mean(),'coverage',df[(df.w==w)&(df.h==1)].n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
 for h in [5,10]:
  q=df[(df.w==w)&(df.h==h)].ic; print(' H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252))
 q=df[(df.w==w)&(df.h==1)].copy()
 for a,b in [('2020','2022'),('2023','2024'),('2025','2026-09-10')]:
  x=q[(q.date>=a)&(q.date<=b)].ic; print(' regime',a,b,'n',len(x),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252) if len(x)>1 else np.nan)
 f=-R.rolling(w,min_periods=max(8,int(w*.75))).std(); print(' turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('dates',P.index.min(),P.index.max(),'assets',P.shape[1])
