import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent/stock_data'; ds={}
for s in U:
 d=pd.read_csv(f'{root}/{s}.csv'); d.date=pd.to_datetime(d.date).dt.normalize(); ds[s]=d.drop_duplicates('date').set_index('date').sort_index()
rows=[]
for s,d in ds.items():
 c=d.close.astype(float); v=d.volume.astype(float).replace(0,np.nan)
 # lagged short reversal strengthened by unusual volume, normalized by volatility
 r=c.pct_change(2); rv=c.pct_change().rolling(20,min_periods=15).std(); vz=np.log(v).diff().rolling(20,min_periods=10).apply(lambda x:(x.iloc[-1]-x.mean())/(x.std()+1e-12),raw=False)
 f=(-r/(rv+1e-12)*(1+vz.clip(lower=0).fillna(0)*0.25)).shift(1); fr=c.shift(-1)/c-1
 rows += [(dt,s,f.loc[dt],fr.loc[dt]) for dt in c.index if pd.notna(f.loc[dt]) and pd.notna(fr.loc[dt])]
q=pd.DataFrame(rows,columns=['date','asset','f','fr'])
def stats(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:z.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 z=pd.Series(z);return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
print('dates',q.date.nunique(),'rows',len(q),'coverage',len(q)/(q.date.nunique()*15));print('daily',stats(q))
for h in [5,10,20]:
 a=[]
 for s,d in ds.items():
  c=d.close.astype(float); v=d.volume.astype(float).replace(0,np.nan);r=c.pct_change(2);rv=c.pct_change().rolling(20,min_periods=15).std();vz=np.log(v).diff().rolling(20,min_periods=10).apply(lambda x:(x.iloc[-1]-x.mean())/(x.std()+1e-12),raw=False);f=(-r/(rv+1e-12)*(1+vz.clip(lower=0).fillna(0)*.25)).shift(1);fr=c.shift(-h)/c-1
  a += [(dt,s,f.loc[dt],fr.loc[dt]) for dt in c.index if pd.notna(f.loc[dt]) and pd.notna(fr.loc[dt])]
 print('horizon',h,stats(pd.DataFrame(a,columns=['date','asset','f','fr'])))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
print('turnover',q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True).diff().abs().mean().mean());q.to_csv('scripts/miner_1_20270203_volume_reversal_signal.csv',index=False)
