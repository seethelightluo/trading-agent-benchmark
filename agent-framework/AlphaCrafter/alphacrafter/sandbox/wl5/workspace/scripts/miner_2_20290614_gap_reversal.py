import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
opens={}; closes={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d.date=pd.to_datetime(d.date); z=d.set_index('date'); opens[a]=z.open; closes[a]=z.close
op=pd.DataFrame(opens).sort_index(); cl=pd.DataFrame(closes).reindex(op.index); gap=op/cl.shift(1)-1
f=-gap.rolling(3,min_periods=2).mean()
for h in [1,5,10]:
 ic=[]; cov=[]; tr=[]; dates=[]
 for i in range(len(cl)-h):
  x=f.iloc[i]; y=cl.iloc[i+h]/cl.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   ic.append(spearmanr(x[ok].rank(pct=True),y[ok]).statistic); cov.append(ok.mean()); dates.append(cl.index[i])
  if i:
   xo=f.iloc[i-1]; ok2=x.notna()&xo.notna()
   if ok2.sum()>=8: tr.append((x[ok2].rank(pct=True)-xo[ok2].rank(pct=True)).abs().mean())
 a=np.array(ic)
 print(h,'dates',len(a),'assets',len(A),'coverage',np.mean(cov),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.mean(tr))
 for name,lo,hi in [('2020-24','2020','2024-12-31'),('2025-26','2025','2026-12-31'),('2027-29','2027','2029-06-14')]:
  q=np.array([v for dt,v in zip(dates,a) if str(dt)>=lo and str(dt)<=hi])
  if len(q)>2: print(' ',name,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('rows',len(cl),'last',cl.index[-1])
