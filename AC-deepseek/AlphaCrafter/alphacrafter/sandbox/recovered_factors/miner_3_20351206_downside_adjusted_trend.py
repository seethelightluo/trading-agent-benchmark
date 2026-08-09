import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}
px=pd.DataFrame(P); r=px.pct_change()
# downside-risk-adjusted 20d trend, all inputs lagged one day
mom=px.pct_change(20).shift(1)
down=r.where(r<0,0).rolling(40).std().shift(1)
f=(mom/(down+0.01)).clip(-10,10)
res={h:[] for h in [1,5,10,20]}; dates=[]; br=[]
for d in px.index:
 z=f.loc[d]; avail=z.dropna().index
 if len(avail)<8: continue
 dates.append(d);br.append(len(avail))
 for h in res:
  fr=(px.shift(-h).loc[d]/px.loc[d]-1).reindex(avail)
  q=pd.DataFrame({'f':z[avail],'r':fr}).dropna()
  res[h].append(spearmanr(q.f,q.r).statistic if len(q)>=8 else np.nan)
print('idea=20d momentum/downside-vol; dates',len(dates),'mean breadth',np.mean(br),'coverage',f.notna().mean().mean())
for h,x in res.items():
 x=np.asarray(x);x=x[np.isfinite(x)]
 print('H',h,'IC %.5f ICIR %.5f hit %.3f n %d'%(x.mean(),x.mean()/(x.std(ddof=1)+1e-12),np.mean(x>0),len(x)))
 for lo,hi in [('2026-01-01','2030-12-31'),('2031-01-01','2035-12-31')]:
  q=np.array([v for d,v in zip(dates,res[h]) if lo<=str(d.date())<=hi and np.isfinite(v)])
  print(lo,'n',len(q),'IC %.5f ICIR %.5f'%(q.mean() if len(q) else np.nan,q.mean()/(q.std(ddof=1)+1e-12) if len(q)>1 else np.nan))
print('rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
