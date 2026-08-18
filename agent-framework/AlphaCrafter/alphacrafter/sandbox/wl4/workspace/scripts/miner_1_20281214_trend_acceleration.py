import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,2500)
 if d is None or len(d)<300: d=get_index_daily_data(s,2500)
 return d
px={s:load(s) for s in U}
# align close panel
p=pd.concat({s:d.set_index(pd.to_datetime(d.date)).close for s,d in px.items() if d is not None},axis=1).sort_index().ffill()
r=np.log(p).diff()
# trend acceleration: recent 20d annualized return minus prior 40d annualized return, risk normalized
fast=r.rolling(20).sum(); prior=r.shift(20).rolling(40).sum(); vol=r.rolling(30).std()*np.sqrt(252)
f=(fast-prior/2)/vol.replace(0,np.nan)
rows=[]
for h in [1,5,10,20]:
 ic=[]; turnovers=[]; ninst=[]
 for i in range(len(p)-h):
  x=f.iloc[i]; y=(p.iloc[i+h]/p.iloc[i]-1)
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ninst.append(len(z))
 # rank turnover sampled daily
 q=f.rank(axis=1,pct=True); turnovers.append(q.diff().abs().mean(axis=1).dropna().mean())
 a=pd.Series(ic).dropna(); print('horizon',h,'dates',len(a),'avg_n',np.mean(ninst),'IC',a.mean(),'ICIR',a.mean()/a.std() if a.std()>0 else np.nan,'hit',np.mean(a>0),'turnover',turnovers[0])
print('cutoff',p.index.max().date(),'assets',p.shape[1],'dates',len(p))
