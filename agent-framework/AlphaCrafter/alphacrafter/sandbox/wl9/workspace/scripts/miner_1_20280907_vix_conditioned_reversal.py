import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for root in ['../persistent/stock_data/','../persistent/index_data/']:
  f=root+s+'.csv'
  if os.path.exists(f): return pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close.astype(float)
D={s:load(s) for s in U};D={s:v for s,v in D.items() if v is not None};P=pd.DataFrame(D).sort_index().ffill();R=np.log(P).diff();v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.reindex(P.index).ffill(); active=v>v.rolling(60,min_periods=40).median();F=-R.rolling(5,min_periods=5).sum().mul(active,axis=0)
print('assets',len(D),'dates',len(P),'active_fraction',round(active.mean(),4))
for h in [1,5,10,20]:
 fw=np.log(P.shift(-h)/P);a=[]
 for dt in F.index:
  if active.loc[dt]:
   z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
   if len(z)>=8:a.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 r=pd.DataFrame(a,columns=['date','ic','n']).set_index('date');q=r.ic
 print('horizon',h,'dates',len(q),'nmean',round(r.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 for nm,x in [('recent252',q.tail(252)),('2027+',q.loc['2027-01-01':]),('2028YTD',q.loc['2028-01-01':])]:
  print(nm,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
