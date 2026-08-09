import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None: continue
 d=d.sort_values('date').reset_index(drop=True); c=d.close.astype(float); o=d.open.astype(float)
 # gap pressure: mean open vs prior close, with sign inverted for next-day reversal
 gap=o/c.shift(1)-1
 # use only completed day's gap, factor predicts next close return
 f=-gap.rolling(5,min_periods=4).mean()
 fr=c.shift(-1)/c-1
 for dt,x,y in zip(d.date,f,fr):
  if np.isfinite(x) and np.isfinite(y): rows.append((dt,s,x,y))
x=pd.DataFrame(rows,columns=['date','s','f','y'])
ics=[]; turns=[]
for dt,g in x.groupby('date'):
 if len(g)>=8:
  ics.append(g.f.corr(g.y)); turns.append(g.set_index('s').f.rank().corr(g.set_index('s').f.rank()))
a=pd.Series(ics).dropna(); print('dates',len(a),'names avg',x.groupby('date').s.nunique().mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'coverage',len(x)/(len(U)*x.date.nunique()))
for h in [1,3,5,10]:
 z=[]
 for s,g in x.groupby('s'):
  # recompute horizon using source rows impossible y is 1d; approximate forward compounded from raw unavailable
  pass
# regime
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=a[[str(i)[:4]>=lo and str(i)[:4]<=hi for i in x.date.unique()]] if False else None
 print(lo,hi)
print('years',[(yr, round(x[x.date.dt.year==yr].groupby('date').apply(lambda g:g.f.corr(g.y)).mean(),4),len(x[x.date.dt.year==yr].groupby('date'))) for yr in sorted(x.date.dt.year.unique())])
# rank turnover across consecutive dates
p=x.pivot_table(index='date',columns='s',values='f').rank(axis=1,pct=True)
print('rank turnover',p.diff().abs().mean().mean())
print('corr proxy existing reversal',x.f.corr(-x.y))
