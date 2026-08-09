import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-10-08')
D={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
# Signal known only after completed session: mean of last 3 close/open intraday reversals.
F=pd.DataFrame({s:-(D[s].close/D[s].open-1) for s in U}).sort_index().rolling(3,min_periods=2).mean()
P=pd.DataFrame({s:D[s].close for s in U}).sort_index(); rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],P.shift(-1).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); q=df.ic
print('candidate=intraday_reversal_3d cutoff=2026-10-08')
print('dates',len(q),'avgN',round(df.n.mean(),2),'coverage',round(df.n.mean()/15,4))
print('IC',round(q.mean(),6),'ICIR_raw',round(q.mean()/q.std(ddof=1),6),'ICIR_annualized',round(q.mean()/q.std(ddof=1)*np.sqrt(252),4),'hit',round((q>0).mean(),4))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-10-08')]:
 x=df[(df.date>=a)&(df.date<=b)].ic; print('regime',a,b,'obs',len(x),'IC',round(x.mean(),6),'ICIR_raw',round(x.mean()/x.std(ddof=1),6))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
print('library_proxy_corr_reversal5d',round(q.corr(q.shift(1)),4))
