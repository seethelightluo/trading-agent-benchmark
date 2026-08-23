import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r20=p.pct_change(20);r60=p.pct_change(60);vol=p.pct_change().rolling(20).std()*np.sqrt(252)
# relative-strength acceleration: recent trend relative to established 60d trend, risk normalized
f=(r20-r60/3)/(vol+1e-8)
def ev(h):
 v=[];c=[];t=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:v.append(z.f.corr(z.y));c.append(len(z))
  if i:
   q=f.iloc[i].rank(pct=True);q0=f.iloc[i-1].rank(pct=True);t.append((q-q0).abs().mean())
 a=np.array(v);return len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)),float(np.mean(c)),float(np.nanmean(t))
print('data',len(p),len(p.columns),p.index.min(),p.index.max())
for h in [5,10,20]:print('h',h,ev(h))
for yr in sorted(set(p.index.year)):
 a=[]
 for i in range(len(p)-10):
  if p.index[i].year!=yr:continue
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:a.append(z.f.corr(z.y))
 if a:print(yr,len(a),round(float(np.mean(a)),5))
