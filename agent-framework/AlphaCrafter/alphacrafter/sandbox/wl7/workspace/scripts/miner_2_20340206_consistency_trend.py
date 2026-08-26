import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
    try: d=get_index_daily_data(s,days=4000)
    except Exception: d=None
    if d is None or len(d)<250:
      try: d=get_stock_daily_data(s,days=4000)
      except Exception: d=None
    if d is not None: xs[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(xs).sort_index().ffill(); r=p.pct_change(); ret20=p.pct_change(20); pos=r.gt(0).rolling(20).mean()
f=ret20*(2*pos-1)/(r.rolling(20).std()*np.sqrt(252)).replace(0,np.nan)
rows=[]
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i],(p.iloc[i+1]/p.iloc[i]-1)],axis=1).dropna()
 if len(q)>=8: rows.append((p.index[i],len(q),q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),q.iloc[:,0].corr(q.iloc[:,1])))
x=pd.DataFrame(rows,columns=['date','n','ic_s','ic_p']).set_index('date')
print('assets',len(xs),list(xs),'dates',len(x),'mean_n',x.n.mean())
for col in ['ic_s','ic_p']:
 a=x[col].dropna(); print(col,'mean',a.mean(),'std',a.std(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'recent180',a.tail(180).mean(),'recent500',a.tail(500).mean())
for h in [1,5,10,20]:
 rr=p.shift(-h)/p-1; vals=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],rr.iloc[i]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna(); print('H',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std())
print('turnover',f.rank(pct=True).diff().abs().mean().mean())
