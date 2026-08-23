import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,2300)
 if d is None or len(d)<100:d=get_index_daily_data(s,2300)
 return d
px={s:get(s) for s in U}; rets={s:d.sort_values('date').set_index('date').close.pct_change() for s,d in px.items() if d is not None}
r=pd.DataFrame(rets).sort_index(); market=r.mean(axis=1); rows=[]
for s in U:
 if s not in r:continue
 x=r[s]; cov=x.rolling(30).cov(market); var=market.rolling(30).var(); beta=cov/var.replace(0,np.nan)
 resid=x-beta*market; f=-resid.rolling(5).sum()/x.rolling(20).std().replace(0,np.nan)
 for dt,v in f.items():
  if pd.notna(v): rows.append((dt,s,v))
z=pd.DataFrame(rows,columns=['date','symbol','factor']);
def evalh(h):
 fw=[]
 for s,d in px.items():
  if d is None:continue
  q=d.sort_values('date'); c=q.close
  fw += [(q.date.iloc[i],s,c.iloc[i+h]/c.iloc[i]-1) for i in range(len(q)-h)]
 a=z.merge(pd.DataFrame(fw,columns=['date','symbol','fwd']),on=['date','symbol']).dropna(); ic=[]
 for _,g in a.groupby('date'):
  if len(g)>=8:ic.append(g.factor.corr(g.fwd,method='spearman'))
 q=pd.Series(ic).dropna();return len(q),a.groupby('date').size().mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),a.symbol.nunique()
print('dates',z.date.nunique(),'rows',len(z),'symbols',z.symbol.nunique())
for h in [5,10,20,40]:print('h',h,evalh(h))
for a,b in [('2025','2026'),('2027','2029'),('2030','2033')]:
 q=z[(z.date>=a)&(z.date<=b)]; fw=[]
 for s,d in px.items():
  if d is None:continue
  x=d.sort_values('date');fw += [(x.date.iloc[i],s,x.close.iloc[i+10]/x.close.iloc[i]-1) for i in range(len(x)-10)]
 q=q.merge(pd.DataFrame(fw,columns=['date','symbol','fwd']),on=['date','symbol']).dropna();ics=[g.factor.corr(g.fwd,method='spearman') for _,g in q.groupby('date') if len(g)>=8]; print('regime',a,b,len(ics),np.nanmean(ics),np.nanmean(ics)/np.nanstd(ics,ddof=1))
rank=z.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',rank.diff().abs().mean(axis=1).mean(),'coverage',len(z)/(15*z.date.nunique()))
