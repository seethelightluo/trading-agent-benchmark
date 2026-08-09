import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
P={a:pd.to_numeric(get_stock_daily_data(a,5000).set_index('date')['close'],errors='coerce') for a in A}
p=pd.DataFrame(P); p.index=pd.to_datetime(p.index); p=p.sort_index()
r=p.pct_change();
# lagged 10-session directional efficiency: net displacement / path length, signed
path=r.shift(1).abs().rolling(10).sum(); net=p.shift(1).pct_change(10).abs(); sign=np.sign(p.shift(1).pct_change(10))
f=(sign*net/path).replace([np.inf,-np.inf],np.nan)
# forward non-overlapping-ish 10 session return
out=[]
for h in [1,5,10,20]:
  fr=p.shift(-h)/p-1
  ics=[]; ns=[]; dates=[]
  for d in f.index:
    x=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
    if len(x)>=8:
      ics.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'));ns.append(len(x));dates.append(d)
  z=pd.Series(ics).replace([np.inf,-np.inf],np.nan).dropna(); ic=z.mean(); ir=ic/z.std(ddof=1) if len(z)>1 else np.nan
  print('H',h,'dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,(z>0).mean()))
  if h==10: print('REG',[(q,len(z[z.index%1==0])) for q in []])
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
# partitions recompute by date labels
h=10;fr=p.shift(-h)/p-1
for name,mask in [('2020-23',f.index<'2024-01-01'),('2024-27',(f.index>='2024-01-01')&(f.index<'2028-01-01')),('2028+',f.index>='2028-01-01'),('latest120',f.index>=f.index.max()-pd.Timedelta(days=180))]:
 z=[]
 for d in f.index[mask]:
  x=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(x)>=8:z.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'))
 z=pd.Series(z).dropna();print(name,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)) if len(z)>1 else 'NA')
