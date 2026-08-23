import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s,n=2300):
 d=get_stock_daily_data(s,n)
 if d is None or len(d)<100: d=get_index_daily_data(s,n)
 return d
px={s:get(s) for s in U}; rows=[]
for s,d in px.items():
 if d is None: continue
 d=d.copy().sort_values('date'); r=d.close.pct_change()
 # 20d trend divided by downside deviation, with a mild persistence term.
 dn=r.where(r<0,0).rolling(30).std(); trend=d.close.pct_change(20)
 persistence=(r.rolling(20).mean()/r.rolling(20).std()).replace([np.inf,-np.inf],np.nan)
 f=(trend/dn.replace(0,np.nan))*(1+0.25*np.tanh(persistence))
 for i in range(35,len(d)-10): rows.append((d.date.iloc[i],s,f.iloc[i],d.close.iloc[i+10]/d.close.iloc[i]-1))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna()
def stats(z):
  ics=[]
  for _,g in z.groupby('date'):
   if len(g)>=8: ics.append(g.factor.corr(g.fwd,method='spearman'))
  q=pd.Series(ics).dropna(); return len(q),z.groupby('date').size().mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),z.symbol.nunique()
print('overall',stats(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]: print('regime',a,b,stats(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 rr=[]
 for s,d in px.items():
  if d is None: continue
  d=d.copy().sort_values('date'); r=d.close.pct_change(); dn=r.where(r<0,0).rolling(30).std(); f=(d.close.pct_change(20)/dn.replace(0,np.nan))*(1+0.25*np.tanh(r.rolling(20).mean()/r.rolling(20).std()))
  for i in range(35,len(d)-h): rr.append((d.date.iloc[i],s,f.iloc[i],d.close.iloc[i+h]/d.close.iloc[i]-1))
 z=pd.DataFrame(rr,columns=['date','symbol','factor','fwd']).dropna(); print('decay',h,stats(z)[0:4])
p=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean(axis=1).mean(),'coverage',len(x)/(len(px)*len(x.date.unique())))
