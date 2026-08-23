import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root='../persistent/stock_data'; ds={}
for s in U:
 d=pd.read_csv(f'{root}/{s}.csv'); d.date=pd.to_datetime(d.date).dt.normalize(); ds[s]=d.drop_duplicates('date').set_index('date').sort_index()
# Cross-asset residual momentum: lagged 10d return minus contemporaneous universe median, volatility scaled
px=pd.DataFrame({s:d.close.astype(float) for s,d in ds.items()}).sort_index(); ret=px.pct_change(10); med=ret.median(axis=1); vol=px.pct_change().rolling(20,min_periods=15).std(); f=(ret.sub(med,axis=0)/(vol+1e-12)).shift(1)
fr=px.shift(-1)/px-1
rows=[]
for date in f.index:
 for s in U:
  if pd.notna(f.loc[date,s]) and pd.notna(fr.loc[date,s]): rows.append((date,s,f.loc[date,s],fr.loc[date,s]))
q=pd.DataFrame(rows,columns=['date','asset','f','fr'])
def calc(x):
 vals=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(vals); return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
print('dates',q.date.nunique(),'rows',len(q),'assets',len(U),'coverage',len(q)/(q.date.nunique()*15))
print('daily',calc(q))
for h in [5,10,20]:
 ff=f
 rr=px.shift(-h)/px-1; a=[]
 for date in ff.index:
  for s in U:
   if pd.notna(ff.loc[date,s]) and pd.notna(rr.loc[date,s]): a.append((date,s,ff.loc[date,s],rr.loc[date,s]))
 print('horizon',h,calc(pd.DataFrame(a,columns=['date','asset','f','fr'])))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,calc(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=f.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
out='scripts/miner_1_20270203_residual_momentum_signal.csv'; q.to_csv(out,index=False); print('signal',out)
