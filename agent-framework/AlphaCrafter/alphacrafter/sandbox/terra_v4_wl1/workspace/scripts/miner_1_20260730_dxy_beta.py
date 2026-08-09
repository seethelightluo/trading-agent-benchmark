import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 except Exception as e: print('missing',s)
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date').set_index('date')
dr=macro.close.pct_change()
# Factor: negative rolling beta to DXY, 60 sessions; current completed date, next kth return
for k in [1,5,10]:
 rec=[]
 for s,x in D.items():
  r=x.close.pct_change(); z=pd.concat([r,dr],axis=1,join='inner').dropna(); z.columns=['r','m']
  cov=z.r.rolling(60,min_periods=45).cov(z.m); var=z.m.rolling(60,min_periods=45).var(); f=-cov/(var+1e-12)
  for i,dt in enumerate(z.index):
   if pd.notna(f.loc[dt]):
    # next k aligned asset observations
    ix=x.index.get_indexer([dt])[0]
    if ix+k<len(x): rec.append((dt,s,float(f.loc[dt]),float(x.close.iloc[ix+k]/x.close.iloc[ix]-1)))
 a=pd.DataFrame(rec,columns=['date','s','f','y']); ic=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:ic.append(spearmanr(g.f,g.y).statistic)
 ic=np.array(ic); print(k,'dates',len(ic),'names',a.groupby('date').size().mean(),'coverage',a.s.nunique()/15,'IC %.6f ICIR %.6f hit %.4f turnoverNA'%(np.nanmean(ic),np.nanmean(ic)/(np.nanstd(ic,ddof=1)+1e-12),np.mean(ic>0)))
# regime split daily
print('regimes')
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=a[(a.date>=lo)&(a.date<=hi+'-12-31')]; z=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic)
 z=np.array(z); print(lo,hi,len(z),np.mean(z),np.mean(z)/(np.std(z,ddof=1)+1e-12))
