import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d.date).dt.normalize(); D[s]=d.drop_duplicates('date').set_index('date')
rows=[]
for s,d in D.items():
 r=d.close.pct_change(); f=-(r.rolling(20,min_periods=20).std()/r.rolling(60,min_periods=60).std()-1); fr=d.close.shift(-1)/d.close-1
 z=pd.DataFrame({'f':f,'fr':fr}); z['symbol']=s; z=z.reset_index(); rows.append(z)
x=pd.concat(rows,ignore_index=True); cs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['f','fr'])
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: cs.append((dt,g.f.corr(g.fr),len(g)))
c=pd.DataFrame(cs,columns=['date','ic','n']).set_index('date'); print('dates',len(c),'avg_names',c.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(c.ic.mean(),c.ic.mean()/c.ic.std(ddof=1),(c.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=c.loc[a:b].ic; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [1,5,10]:
 rr=[]
 for s,d in D.items():
  r=d.close.pct_change(); f=-(r.rolling(20,min_periods=20).std()/r.rolling(60,min_periods=60).std()-1); fr=d.close.shift(-h)/d.close-1
  rr.append(pd.DataFrame({'date':d.index,'f':f.values,'fr':fr.values}))
 y=pd.concat(rr,ignore_index=True); out=[]
 for dt,g in y.groupby('date'):
  g=g.dropna()
  if len(g)>=8 and g.f.nunique()>1: out.append(g.f.corr(g.fr))
 out=pd.Series(out).dropna(); print('h',h,'dates',len(out),'IC',out.mean(),'ICIR',out.mean()/out.std(ddof=1))
wide=x.pivot(index='date',columns='symbol',values='f'); print('turnover',wide.rank(pct=True).diff().abs().mean().mean())
