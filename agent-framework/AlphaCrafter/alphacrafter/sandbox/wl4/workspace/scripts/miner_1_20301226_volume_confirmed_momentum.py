import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; end=pd.Timestamp('2030-12-25')
d={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float) for s,x in d.items()}).sort_index().loc[:end].ffill()
V=pd.DataFrame({s:x.volume.astype(float) for s,x in d.items()}).sort_index().loc[C.index].replace(0,np.nan).ffill()
r=C.pct_change()
# Volume-confirmed momentum: medium-term return weighted by standardized log-volume surprise.
# Cross-sectional rank of the lagged signal avoids scale differences across asset classes.
ret=C.pct_change(20)
logv=np.log(V)
vs=(logv-logv.rolling(60,min_periods=30).mean())/logv.rolling(60,min_periods=30).std()
raw=ret*(1+0.25*vs.clip(-2,2))
sig=raw.rank(axis=1,pct=True).shift(1)
for h in [5,10,20]:
 rows=[]
 for i in range(len(C)-h):
  z=pd.concat([sig.iloc[i],C.iloc[i+h]/C.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((C.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 a=pd.DataFrame(rows,columns=['date','n','ic']); rec=a[a.date>=a.date.max()-pd.Timedelta(days=365)]
 def met(q): return (q.ic.mean(), q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(len(q)))
 print(f'h{h}: dates={len(a)} avgN={a.n.mean():.2f} IC={met(a)[0]:.6f} ICIR={met(a)[1]:.6f} hit={(a.ic>0).mean():.4f} recentIC={met(rec)[0]:.6f} recentICIR={met(rec)[1]:.6f}')
print(f'coverage={sig.notna().mean().mean():.4f} turnover={sig.diff().abs().mean(axis=1).mean():.6f} instruments={len(U)} cutoff={C.index[-1].date()}')
