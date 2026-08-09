import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-09-10')
D={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
# independent candidate: 3d intraday reversal, volatility normalized
F=pd.DataFrame({s:-(D[s].close/D[s].open-1) for s in U}).sort_index().rolling(3,min_periods=2).mean()
P=pd.DataFrame({s:D[s].close for s in U}).sort_index(); rows=[]
for dt in F.index:
 for h in [1,5,10]:
  z=pd.concat([F.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('candidate=intraday_reversal_3d cutoff=2026-09-10')
print('dates',df[df.h==1].date.nunique(),'avgN',round(df[df.h==1].n.mean(),2),'coverage',round(df[df.h==1].n.mean()/15,4))
for h in [1,5,10]:
 q=df[df.h==h].ic; print('H',h,'obs',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),4),'hit',round((q>0).mean(),4))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-09-10')]:
 q=df[(df.h==1)&(df.date>=a)&(df.date<=b)].ic; print('regime',a,b,'obs',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),4))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
# robust windows
for end in ['2024-12-31','2026-01-01','2026-09-10']:
 q=df[(df.h==1)&(df.date>=pd.Timestamp(end)-pd.Timedelta(days=365))&(df.date<=end)].ic
 print('rolling1y',end,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1)*np.sqrt(252),4))
