import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 d=get_stock_daily_data(a,days=3000)
 if d is not None and len(d)>120: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change(); v=r.rolling(20,min_periods=10).std()
f=(-p.pct_change(3)).where(v.le(v.quantile(.75,axis=1),axis=0))
peer=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
for i in range(len(p)):
 z=r.iloc[i-5] if i>=5 else pd.Series(dtype=float)
 for a in p.columns: peer.iloc[i,peer.columns.get_loc(a)]=z.drop(labels=a,errors='ignore').median()
base={'short5':-p.pct_change(5),'rev3':-p.pct_change(3),'mom20':p.pct_change(20)/r.rolling(20).std(),'peer5':peer}
for n,x in base.items():
 q=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna()
 print(n, len(q), q.f.corr(q.x))
# daily IC full and halves, forward horizons
for h in [1,5,10]:
 vals=[]; dates=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.pct_change(h).iloc[i+h]).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: vals.append(q.f.corr(q.y));dates.append(p.index[i]);ns.append(len(q))
 x=np.array(vals); print('h',h,'obs',len(x),'meanN',np.mean(ns),'IC',x.mean(),'std',x.std(ddof=1),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'early',x[:len(x)//2].mean(),'late',x[len(x)//2:].mean())
# rank turnover among valid set
turn=[]
for i in range(1,len(f)):
 q=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
 if len(q)>=8: turn.append((q.iloc[:,0].rank()!=q.iloc[:,1].rank()).mean())
print('coverage',f.notna().sum().sum()/(f.size),'turnover',np.mean(turn),'dates',len(p),'instruments',len(px))
print('period',p.index.min().date(),p.index.max().date())
