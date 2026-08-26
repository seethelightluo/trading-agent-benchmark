import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(sym):
    x=get_stock_daily_data(sym,days=6000)
    if x is None or len(x)<300: x=get_index_daily_data(sym,days=6000)
    if x is None or len(x)==0:return pd.Series(dtype=float)
    x=x.copy();x['date']=pd.to_datetime(x['date']);return x.set_index('date')['close'].astype(float)
p=pd.DataFrame({s:get(s) for s in U}).sort_index()
lr=np.log(p).diff(5).shift(1);vol=np.log(p).diff().rolling(20).std().shift(1)*np.sqrt(20)
peer=lr.sub(lr.median(axis=1),axis=0);f=-peer/vol;fr=np.log(p).shift(-10)-np.log(p)
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:rows.append((d,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(r),'avgN',r.n.mean(),'coverage',len(r)/(len(p)-10),'IC',r.ic.mean(),'std',r.ic.std(),'ICIR',r.ic.mean()/r.ic.std(),'hit',(r.ic>0).mean())
for label,mask in [('early',r.index<'2025-01-01'),('mid',(r.index>='2025-01-01')&(r.index<'2030-01-01')),('late',r.index>='2030-01-01'),('recent',r.index>=r.index.max()-pd.Timedelta(days=365))]:
 q=r.loc[mask];print(label,len(q),q.ic.mean(),q.ic.mean()/q.ic.std() if len(q)>1 else np.nan,(q.ic>0).mean() if len(q) else np.nan)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for h in [1,5,10,20]:
 frh=np.log(p).shift(-h)-np.log(p);v=[]
 for d in f.index:
  z=pd.concat([f.loc[d],frh.loc[d]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('horizon',h,'n',len(v),'ic',np.nanmean(v),'icir',np.nanmean(v)/np.nanstd(v))
