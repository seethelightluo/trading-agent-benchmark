import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-10-08')
rows=[]; sig=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]
 r=x.close.pct_change(); mom=x.close.pct_change(20); vol=r.rolling(20,min_periods=12).std(); f=mom/vol.replace(0,np.nan)
 y=x.close.shift(-1)/x.close-1
 q=pd.DataFrame({'f':f,'y':y}).dropna()
 for d,z in q.iterrows(): rows.append((d,s,z.f,z.y))
 for d,v in f.dropna().items(): sig.append({'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)})
a=pd.DataFrame(rows,columns=['date','symbol','f','y']); obs=[]
for d,g in a.groupby('date'):
 if len(g)>=8:
  c=g.f.corr(g.y,method='spearman')
  if pd.notna(c): obs.append((d,c,len(g)))
z=pd.DataFrame(obs,columns=['date','ic','n'])
def stat(v): return (v.mean(),v.mean()/v.std(ddof=1),(v>0).mean())
print('cutoff',C.date(),'dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(len(a)/(len(z)*15),4))
print('daily IC ICIR hit',*[round(x,6) for x in stat(z.ic)])
for yr,g in z.groupby(z.date.dt.year): print('regime',yr,'dates',len(g),'IC ICIR',round(g.ic.mean(),6),round(g.ic.mean()/g.ic.std(ddof=1),6))
for h in [5,10]:
 vals=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]; r=x.close.pct_change(); f=x.close.pct_change(20)/r.rolling(20,min_periods=12).std(); y=x.close.shift(-h)/x.close-1
  q=pd.DataFrame({'f':f,'y':y}).dropna()
  for d,v in q.groupby(q.index): pass
  vals += [(d,g.f.corr(g.y,method='spearman')) for d,g in pd.DataFrame({'f':f,'y':y}).dropna().groupby(level=0) if False]
 # recompute pooled rows by date
 oo=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]; r=x.close.pct_change(); f=x.close.pct_change(20)/r.rolling(20,min_periods=12).std(); y=x.close.shift(-h)/x.close-1
  q=pd.DataFrame({'f':f,'y':y}).dropna(); oo += [(d,s,row.f,row.y) for d,row in q.iterrows()]
 b=pd.DataFrame(oo,columns=['date','s','f','y']); vv=[]
 for d,g in b.groupby('date'):
  if len(g)>=8:
   c=g.f.corr(g.y,method='spearman')
   if pd.notna(c): vv.append(c)
 vv=np.array(vv); print('horizon',h,'dates',len(vv),'IC ICIR',round(vv.mean(),6),round(vv.mean()/vv.std(ddof=1),6))
print('signal_rows',len(sig),'symbols',a.symbol.nunique()); pd.DataFrame(sig).to_csv('scripts/miner_2_20261008_risk_adj_momentum_signal.csv',index=False)
