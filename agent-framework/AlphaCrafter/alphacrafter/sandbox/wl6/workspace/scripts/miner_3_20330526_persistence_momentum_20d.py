import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(sym,n=2300):
 f=get_stock_daily_data(sym,n)
 if f is None or len(f)<100: f=get_index_daily_data(sym,n)
 return f
px={s:get(s) for s in U}; rows=[]
for s,d in px.items():
 if d is None: continue
 d=d.copy(); d['date']=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); r=d.close.pct_change(); vol=r.rolling(20).std(); p=2*r.gt(0).rolling(40).mean()-1; f=d.close.pct_change(20)/vol.replace(0,np.nan)*p
 for i in range(60,len(d)-10): rows.append((d.index[i],s,f.iloc[i],d.close.iloc[i+10]/d.close.iloc[i]-1))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna()
def daily(df, col='factor', y='fwd'):
 out=[]
 for _,g in df.groupby('date'):
  if len(g)>=8: out.append(g[col].corr(g[y],method='spearman'))
 return pd.Series(out).dropna()
ics=daily(x); r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); turn=r.diff().abs().mean(axis=1).dropna()
print({'dates':len(ics),'avg_instruments':float(x.groupby('date').size().mean()),'ic':float(ics.mean()),'icir':float(ics.mean()/ics.std(ddof=1)),'hit':float((ics>0).mean()),'coverage':float(np.mean([n/15 for n in x.groupby('date').size()])),'turnover':float(turn.mean())})
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]:
 z=daily(x[(x.date>=a)&(x.date<=b)]); print('regime',a,b,len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)) if len(z)>1 else np.nan)
for h in [5,10,20,40]:
 rr=[]
 for s,d in px.items():
  if d is None: continue
  d=d.copy().sort_values('date'); ret=d.close.pct_change(); vol=ret.rolling(20).std(); p=2*ret.gt(0).rolling(40).mean()-1; f=d.close.pct_change(20)/vol.replace(0,np.nan)*p
  for i in range(60,len(d)-h): rr.append((d.date.iloc[i],s,f.iloc[i],d.close.iloc[i+h]/d.close.iloc[i]-1))
 z=daily(pd.DataFrame(rr,columns=['date','s','factor','fwd']).dropna()); print('decay',h,len(z),float(z.mean()))
