import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
C=pd.DataFrame({s:d.close for s,d in D.items()}); O=pd.DataFrame({s:d.open for s,d in D.items()}); H=pd.DataFrame({s:d.high for s,d in D.items()}); L=pd.DataFrame({s:d.low for s,d in D.items()}); R=(H-L).replace(0,np.nan)
# factor directions are reversal: positive expected forward return
V={'body_rev':-(C-O)/R,'range_rev':-(2*(C-L)/R-1),'upper_tail_rev':-((H-C)/R-(C-L)/R),'body_range_ratio':-(C-O).abs()/R*np.sign(C-O), 'close_to_open_gap':-(O/C.shift(1)-1)}
for nm,F in V.items():
 vals=[]; ns=[]; ds=[]
 for dt in F.index:
  xs=[];ys=[]
  for s in U:
   if pd.isna(F.loc[dt,s]) or dt not in D[s].index: continue
   ix=D[s].index.get_loc(dt)
   if ix+1<len(D[s]): xs.append(F.loc[dt,s]);ys.append(D[s].iloc[ix+1].close/D[s].iloc[ix].close-1)
  if len(xs)>=8 and len(set(xs))>1: vals.append(spearmanr(xs,ys).statistic);ns.append(len(xs));ds.append(dt)
 a=np.array(vals); print(nm,len(a),round(np.mean(ns),2),round(a.mean(),5),round(a.mean()/a.std(ddof=1),5),round((a>0).mean(),4),'cov',round(len(a)/len(F),3))
 # years
 for y in [2020,2021,2022,2023,2024,2025,2026]:
  z=a[[d.year==y for d in ds]]
  if len(z): print(y,round(z.mean(),4),end=';')
 print()
 # correlation with existing clv/reversal pooled
 clv=-(2*(C-L)/R-1); rev=-(C/C.shift(5)-1)
 z=pd.concat([F.stack(),clv.stack(),rev.stack()],axis=1).dropna(); print('corr',z.corr(method='spearman').iloc[0].round(3).tolist())
