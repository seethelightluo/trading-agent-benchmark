import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2033-06-08')
px={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
for s in U:px[s]=px[s][px[s].index<=end]
dates=sorted(set.intersection(*[set(v.index) for v in px.values()]))
def calc(dt,h):
 f={}; y={}
 for s,x in px.items():
  z=x.loc[:dt]; fut=x[x.index>dt].head(h)
  if len(z)<32 or len(fut)<h: continue
  rr=z.pct_change().iloc[-21:].dropna(); down=rr[rr<0]
  # reward low downside volatility, adjusted by medium trend
  if len(down)>=3 and down.std()>0:
   f[s]=-(down.std()*np.sqrt(252)) + 0.25*(z.iloc[-1]/z.iloc[-21]-1)
   y[s]=fut.iloc[-1]/x.loc[dt]-1
 c=set(f)&set(y)
 return (spearmanr([f[s] for s in c],[y[s] for s in c]).statistic,len(c)) if len(c)>=8 else (np.nan,0)
for h in [1,5,10,20]:
 a=[]; ns=[]
 for dt in dates:
  q,n=calc(dt,h)
  if n>=8:a.append(q);ns.append(n)
 print(h,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(np.array(a)>0),np.mean(ns))
# 10d detailed
rows=[]
for dt in dates:
 q,n=calc(dt,10)
 if n>=8:rows.append((dt,q,n))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('coverage',r.n.mean()/15)
for label,lo,hi in [('pre','2020-01-01','2029-12-31'),('post','2030-01-01','2033-06-08'),('recent',end-pd.Timedelta(days=365),end)]:
 q=r[(r.index>=pd.Timestamp(lo))&(r.index<=pd.Timestamp(hi))].ic;print(label,len(q),q.mean(),q.mean()/q.std(ddof=1))
r.reset_index().to_csv('scripts/miner_2_20330609_downside_quality_ic.csv',index=False)
