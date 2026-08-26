import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for root in ['../persistent/stock_data/','../persistent/index_data/']:
  f=root+s+'.csv'
  if os.path.exists(f): return pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close.astype(float)
D={s:load(s) for s in U}; D={s:v for s,v in D.items() if v is not None}; P=pd.DataFrame(D).sort_index().ffill(); R=np.log(P).diff()
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().close.reindex(P.index).ffill()
active=(macro>macro.rolling(60,min_periods=40).median())
F=-R.rolling(5,min_periods=5).sum().mul(active,axis=0); rows=[]
for h in [1,5,10,20]:
 fw=np.log(P.shift(-h)/P); rows=[]
 for dt in F.index:
  if active.loc[dt]:
   z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
   if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
 print('horizon',h,'dates',len(r),'nmean',round(r.n.mean(),2),'coverage',round(F.loc[active].notna().mean().mean(),4),'IC',round(r.ic.mean(),6),'ICIR',round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4))
 for nm,q in [('recent252',r.tail(252)),('2027+',r.loc['2027-01-01':]),('2028YTD',r.loc['2028-01-01':])]:
  if len(q)>1: print(nm,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6))
print('active_fraction',round(active.mean(),4),'assets',len(D),'dates',len(P))
