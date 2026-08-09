import pandas as pd, numpy as np, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.sort_index()
print('assets',len(D))
spx=D['SPX'].pct_change(); rows=[]
for s,p in D.items():
 ret=p.pct_change(); beta=ret.rolling(60).cov(spx)/spx.rolling(60).var(); z=pd.DataFrame({'f':p.pct_change(20)-beta*spx.pct_change(20),'f2':p.pct_change(20)-spx.pct_change(20),'y':p.shift(-1)/p-1})
 for dt,r in z.loc[:'2026-07-15'].dropna().iterrows(): rows.append((dt,s,float(r.f),float(r.f2),float(r.y)))
x=pd.DataFrame(rows,columns=['date','s','f','f2','y'])
def calc(q, col, y='y'):
 vals=[]
 for _,g in q.groupby('date'):
  if len(g)>=8: vals.append(g[col].corr(g[y]))
 return pd.Series(vals).dropna()
for col in ['f','f2']:
 cs=calc(x,col); print(col,'dates',len(cs),'avgN',x.groupby('date').size().mean(),'IC',round(cs.mean(),6),'ICIR',round(cs.mean()/cs.std(ddof=1),6),'hit',round((cs>0).mean(),4))
 for h in [5,10]:
  yy=pd.concat({s:p.shift(-h)/p-1 for s,p in D.items()},axis=1).stack().rename('yy').reset_index(); yy.columns=['date','s','yy']; q=x[['date','s',col]].merge(yy,on=['date','s']); cs2=calc(q,col,'yy'); print(' ',h,'dates',len(cs2),'IC',round(cs2.mean(),6),'ICIR',round(cs2.mean()/cs2.std(ddof=1),6))
print('date range',x.date.min(),x.date.max(),'instruments',x.s.nunique())
