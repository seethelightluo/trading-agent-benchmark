import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,2300)
 if d is None or len(d)<100: d=get_index_daily_data(s,2300)
 return d
px={s:get(s) for s in U}; rows=[]
for s,d in px.items():
 if d is None: continue
 d=d.sort_values('date').copy(); r=d.close.pct_change(); vol=r.rolling(20).std()
 # Short-horizon reversal, risk scaled and activated only when medium trend is positive.
 # The gate avoids buying persistent losers while retaining mean reversion in established trends.
 trend=d.close.pct_change(60)
 f=(-d.close.pct_change(5)/vol.replace(0,np.nan))*np.where(trend>0,1.0,0.25)
 for i in range(65,len(d)-40): rows.append((d.date.iloc[i],s,f.iloc[i]))
base=pd.DataFrame(rows,columns=['date','symbol','factor']).dropna()
def stats(z,h):
 y=[]
 for s,d in px.items():
  if d is None: continue
  q=d.sort_values('date'); close=q.close
  for i in range(65,len(q)-h): y.append((q.date.iloc[i],s,close.iloc[i+h]/close.iloc[i]-1))
 fw=pd.DataFrame(y,columns=['date','symbol','fwd']); z=z.merge(fw,on=['date','symbol']).dropna(); ics=[]
 for _,g in z.groupby('date'):
  if len(g)>=8: ics.append(g.factor.corr(g.fwd,method='spearman'))
 q=pd.Series(ics).dropna(); return len(q),z.groupby('date').size().mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),z.symbol.nunique()
print('rows',len(base),'dates',base.date.nunique(),'symbols',base.symbol.nunique())
for h in [5,10,20,40]:
 print('horizon',h,stats(base,h))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]: print('regime',a,b,stats(base[(base.date>=a)&(base.date<=b)],10))
rank=base.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean(),'coverage',len(base)/(15*base.date.nunique()))
