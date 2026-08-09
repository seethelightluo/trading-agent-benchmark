import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-08-27'); base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date')['close'].astype(float)
 px[s]=d[d.index<=cut]
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Negative realized downside asymmetry: negative of 20-session return skewness; completed data only
f=-R.rolling(20,min_periods=15).skew()
rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('date range',P.index.min(),P.index.max())
q=df[df.h==1]; print('daily dates',q.date.nunique(),'obs',len(q),'avg names',q.n.mean(),'coverage',q.n.mean()/15,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252),'hit',(q.ic>0).mean())
for h in [1,5,10]:
 q=df[df.h==h]; print('H',h,'obs',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252),'hit',(q.ic>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026-08-27')]:
 q=df[(df.h==1)&(df.date>=a)&(df.date<=b)].ic; print('regime',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
r=f.rank(axis=1,pct=True); print('turnover',((r-r.shift()).abs().mean(axis=1)).mean())
