import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index() for s in U}
macro={m:pd.read_csv('../persistent/index_data/'+m+'.csv',parse_dates=['date']).set_index('date').sort_index().close for m in ['DXY','VIX']}
def factor(m):
 out={}
 for s,d in D.items():
  p=d.close; mr=macro[m].reindex(p.index).ffill(); ar=p.pct_change(); br=mr.pct_change()
  out[s]=-(ar.rolling(60,min_periods=40).cov(br)/br.rolling(60,min_periods=40).var())
 return out
for m in ['DXY','VIX']:
 F=factor(m); print('\nMACRO',m)
 for h in [1,5,10]:
  vals=[];ns=[]
  dates=sorted(set().union(*[x.index for x in F.values()]))
  for dt in dates:
   if dt>end: continue
   x=[];y=[]
   for s in U:
    if dt not in F[s].index or pd.isna(F[s].loc[dt]):continue
    d=D[s]; i=d.index.get_loc(dt); j=i+h
    if j<len(d) and pd.notna(d.iloc[j].close):x.append(F[s].loc[dt]);y.append(d.iloc[j].close/d.iloc[i].close-1)
   if len(x)>=8 and np.std(x)>0: vals.append(pd.Series(x).corr(pd.Series(y),method='spearman'));ns.append(len(x))
  a=np.array(vals); print(h,'dates',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 valid=[dt for dt in sorted(set().union(*[x.index for x in F.values()])) if dt<=end and sum(dt in F[s].index and pd.notna(F[s].loc[dt]) for s in U)>=8]
 print('coverage',len(valid)/len([dt for dt in sorted(set().union(*[x.index for x in F.values()])) if dt<=end]),'turnover',np.mean([pd.Series({s:F[s].loc[dt] for s in U if dt in F[s].index and pd.notna(F[s].loc[dt])}).rank(pct=True).sub(pd.Series({s:F[s].loc[prev] for s in U if prev in F[s].index and pd.notna(F[s].loc[prev])}).rank(pct=True)).abs().mean() for prev,dt in zip(valid[:-1],valid[1:]) if sum(s in F and prev in F[s].index and dt in F[s].index for s in U)>=8]))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
  z=[]
  for dt in valid:
   if not (pd.Timestamp(lo+'-01-01')<=dt<=pd.Timestamp(hi+'-12-31')):continue
   x=[];y=[]
   for s in U:
    if dt in F[s].index and pd.notna(F[s].loc[dt]):
     d=D[s];i=d.index.get_loc(dt)
     if i+1<len(d):x.append(F[s].loc[dt]);y.append(d.iloc[i+1].close/d.iloc[i].close-1)
   if len(x)>=8:z.append(pd.Series(x).corr(pd.Series(y),method='spearman'))
  print('regime',lo,hi,len(z),np.mean(z))
 # library correlations pooled
 for nm,X in [('rev5',lambda d:-(d.close/d.close.shift(5)-1)),('mom20',lambda d:d.close/d.close.shift(20)-1),('ram20',lambda d:(d.close/d.close.shift(20)-1)/d.close.pct_change().rolling(60).std())]:
  a=[];b=[]
  for s in U:
   q=pd.concat([F[s],X(D[s])],axis=1).loc[:end].dropna();a+=q.iloc[:,0].tolist();b+=q.iloc[:,1].tolist()
  print('corr',nm,pd.Series(a).corr(pd.Series(b),method='spearman'))
