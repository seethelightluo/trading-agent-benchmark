import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); D[s]=x
 except Exception as e: print('missing',s,e)
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:'2026-07-15']
d=macro.close.pct_change()
# FX-beta defensive: negative rolling 40d covariance with DXY, standardized by asset vol and DXY variance
rows=[]
for s,x in D.items():
 r=x.close.pct_change(); z=pd.concat([r,d],axis=1,join='inner'); z.columns=['r','m']
 beta=z.r.rolling(40,min_periods=30).cov(z.m)/z.m.rolling(40,min_periods=30).var()
 # negative beta means benefits from dollar strength; factor rewards lower beta (negative beta larger)
 f=-beta
 for i,dt in enumerate(z.index):
  if pd.notna(f.loc[dt]) and i+1<len(z): rows.append((dt,s,float(f.loc[dt]),float(z.r.iloc[i+1])))
a=pd.DataFrame(rows,columns=['date','s','f','y']); ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ics.append(spearmanr(g.f,g.y).statistic)
ics=np.array(ics)
print('idea=negative DXY beta 40d; valid_dates',len(ics),'avg_names',a.groupby('date').size().mean(),'assets',a.s.nunique(),'coverage',a.s.nunique()/15)
print('daily IC %.8f ICIR %.8f hit %.5f turnover unavailable'%(np.nanmean(ics),np.nanmean(ics)/(np.nanstd(ics,ddof=1)+1e-12),np.mean(ics>0)))
for k in [5,10]:
 rr=[]
 for s,x in D.items():
  r=x.close.pct_change(); z=pd.concat([r,d],axis=1,join='inner'); z.columns=['r','m']; f=-(z.r.rolling(40,min_periods=30).cov(z.m)/z.m.rolling(40,min_periods=30).var())
  for i,dt in enumerate(z.index):
   if pd.notna(f.iloc[i]) and i+k<len(z):rr.append((dt,float(f.iloc[i]),float(z.r.iloc[i+1:i+k+1].sum())))
 q=pd.DataFrame(rr,columns=['date','f','y']); ic=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:ic.append(spearmanr(g.f,g.y).statistic)
 ic=np.array(ic);print('%dd IC %.8f ICIR %.8f dates %d'%(k,np.nanmean(ic),np.nanmean(ic)/(np.nanstd(ic,ddof=1)+1e-12),len(ic)))
