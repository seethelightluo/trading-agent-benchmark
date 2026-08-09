import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); D={}
for s in U:
 p=Path('../persistent/stock_data')/(s+'.csv'); x=pd.read_csv(p); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index(); bench=R.mean(axis=1); out=[]
# stress resilience: mean asset return on broad negative days minus its normal mean, normalized by downside vol
for w in [40,60,120]:
 stress=bench < 0
 n=R.where(stress).rolling(w,min_periods=max(15,w//3)).mean()
 allm=R.rolling(w,min_periods=max(15,w//3)).mean()
 down=R.where(R<0).rolling(w,min_periods=max(15,w//3)).std()
 F=(n-allm)/down
 for h in [1,5,10]:
  Y=R.shift(-1).rolling(h).sum().shift(-(h-1)); vals=[]; nms=[]
  for dt in R.index:
   z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic); nms.append(len(z))
  q=pd.Series(vals).dropna(); out.append((w,h,q.mean(),q.mean()/q.std(ddof=1), (q>0).mean(),len(q),np.mean(nms)))
for a in out: print('w,h %d,%d IC %.5f ICIR %.5f hit %.3f dates %d avg_names %.2f'%a)
print('range',R.index.min(),R.index.max(),'assets',len(U))
