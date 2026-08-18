import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close
P=pd.DataFrame(px).sort_index().ffill(); r=np.log(P).diff()
# lagged 30-session return divided by downside deviation, rewarding returns achieved with limited downside
neg=r.where(r<0); down=neg.shift(1).rolling(30,min_periods=15).std()
f=(P.shift(1)/P.shift(31)-1)/(down*np.sqrt(252)+1e-12)
y=P.shift(-10)/P-1
ics=[]; ns=[]; turns=[]; prev=None
for dt in f.index:
 a,b=f.loc[dt],y.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: ics.append(a[ok].corr(b[ok],method='spearman')); ns.append(ok.sum())
 z=a.rank(pct=True)
 if prev is not None:
  q=z.notna()&prev.notna(); turns.append((z[q]-prev[q]).abs().mean())
 prev=z
ic=pd.Series(ics).dropna(); print('candidate=downside_adjusted_mom30 dates',len(ic),'avg_names',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,4)); print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'turnover',round(np.mean(turns),4))
for k in [120,252,504]:
 z=ic.tail(k); print('recent',k,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
for i,z in enumerate(np.array_split(ic,4)): print('block',i+1,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('decay')
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; vv=[]
 for dt in f.index:
  a,b=f.loc[dt],yy.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: vv.append(a[ok].corr(b[ok],method='spearman'))
 print(h,round(np.mean(vv),6),len(vv))
# artifact
f.to_csv('factors/miner_1_20350302_downside_adjusted_mom30_signal.csv')
