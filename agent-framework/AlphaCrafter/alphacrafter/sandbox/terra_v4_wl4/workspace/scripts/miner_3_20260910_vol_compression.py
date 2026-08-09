import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-09-09')
rows=[]; sigrows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]
 r=x.close.pct_change(); v5=r.rolling(5,min_periods=5).std(); v20=r.rolling(20,min_periods=15).std()
 # compression: lower recent realized volatility versus medium-term volatility
 f=-np.log((v5+1e-12)/(v20+1e-12)); y=x.close.shift(-1)/x.close-1
 z=pd.DataFrame({'f':f,'y':y}).dropna()
 for d,q in z.iterrows(): rows.append((d,s,float(q.f),float(q.y)))
 for d,v in f.dropna().items(): sigrows.append({'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)})
a=pd.DataFrame(rows,columns=['date','symbol','f','y']); obs=[]
for d,g in a.groupby('date'):
 if len(g)>=8:
  q=g.f.corr(g.y,method='spearman')
  if pd.notna(q): obs.append((d,q,len(g)))
z=pd.DataFrame(obs,columns=['date','ic','n']);
print('cutoff',C.date(),'dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(len(a)/sum(len(pd.read_csv('../persistent/stock_data/'+s+'.csv')) for s in U),4))
print('IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
for h in [1,5,10]:
 rr=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]; r=x.close.shift(-h)/x.close-1; vol=x.close.pct_change().rolling(5,min_periods=5).std()/x.close.pct_change().rolling(20,min_periods=15).std(); f=-np.log(vol+1e-12); q=pd.DataFrame({'f':f,'y':r}).dropna()
  rr += [(d,s,float(v.f),float(v.y)) for d,v in q.iterrows()]
 b=pd.DataFrame(rr,columns=['date','symbol','f','y']); oo=[]
 for d,g in b.groupby('date'):
  if len(g)>=8:
   q=g.f.corr(g.y,method='spearman')
   if pd.notna(q): oo.append(q)
 oo=np.array(oo); print('horizon',h,'dates',len(oo),'IC',round(oo.mean(),6),'ICIR',round(oo.mean()/oo.std(ddof=1),6))
for yr,g in z.groupby(z.date.dt.year): print('regime',yr,'n',len(g),'IC',round(g.ic.mean(),5),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),4))
q=a.sort_values(['symbol','date']).groupby('symbol').f.diff().abs(); print('turnover_proxy',round(q.mean(),5))
pd.DataFrame(sigrows).to_csv('scripts/miner_3_20260910_vol_compression_signal.csv',index=False)
print('signal_rows',len(sigrows),'symbols',a.symbol.nunique())
