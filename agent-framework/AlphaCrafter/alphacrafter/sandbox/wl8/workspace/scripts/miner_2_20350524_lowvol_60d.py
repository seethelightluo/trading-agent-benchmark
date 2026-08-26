import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 x=get_stock_daily_data(s,days=6000)
 if x is None or len(x)<300:x=get_index_daily_data(s,days=6000)
 if x is None or len(x)==0:return pd.Series(dtype=float)
 x=x.copy();x.date=pd.to_datetime(x.date);return x.set_index('date').close.astype(float)
p=pd.DataFrame({s:g(s) for s in U}).sort_index();lr=np.log(p).diff();v=lr.rolling(60).std().shift(1);f=-v
fr=np.log(p).shift(-10)-np.log(p);rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:rows.append((d,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');print('dates',len(r),'avgN',r.n.mean(),'coverage',len(r)/(len(p)-10),'IC',r.ic.mean(),'std',r.ic.std(),'ICIR',r.ic.mean()/r.ic.std(),'hit',(r.ic>0).mean());
for label,m in [('mid',(r.index>='2025-01-01')&(r.index<'2030-01-01')),('late',r.index>='2030-01-01'),('recent',r.index>=r.index.max()-pd.Timedelta(days=365))]:
 q=r[m];print(label,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for h in [1,5,10,20]:
 fh=np.log(p).shift(-h)-np.log(p);a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fh.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('horizon',h,'n',len(a),'ic',np.nanmean(a),'icir',np.nanmean(a)/np.nanstd(a))
