import pandas as pd, numpy as np, glob
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Tail-adjusted momentum: trailing mean return rewarded, downside deviations penalized.
mean20=r.rolling(20,min_periods=15).mean()
down=np.minimum(r,0.0).rolling(20,min_periods=15).apply(lambda x: np.sqrt(np.mean(np.asarray(x)**2)),raw=True)
f=mean20/(down+1e-8)
rows=[]; turns=[]
for i in range(20,len(p)-1):
    x=f.iloc[i]; y=r.iloc[i+1]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    if i>=1:
        a=f.iloc[i-1].rank(); b=x.rank(); turns.append(np.mean(a.notna()&b.notna()&(a!=b)))
a=np.asarray(rows); print('dates',len(a),'names_mean',np.mean([sum(np.isfinite(f.iloc[i])) for i in range(20,len(p)-1)]),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'turn',np.nanmean(turns))
for lo,hi in [(0, len(p)),(0,750),(750,1250),(1250,len(p))]:
 q=a[lo:hi]; print('window',lo,hi,'n',len(q),'icir',np.nanmean(q)/np.nanstd(q,ddof=1) if len(q)>1 else np.nan,'mean',np.nanmean(q))
for h in [1,5,10]:
 yy=p.pct_change(h).shift(-h)
 q=[]
 for i in range(20,len(p)-h):
  z=pd.concat([f.iloc[i],yy.iloc[i]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.asarray(q);print('horizon',h,'n',len(q),'ic',np.nanmean(q),'icir',np.nanmean(q)/np.nanstd(q,ddof=1))
