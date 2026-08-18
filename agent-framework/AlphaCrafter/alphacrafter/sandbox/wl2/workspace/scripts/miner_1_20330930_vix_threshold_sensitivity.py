import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d)>150:return d
  except: pass
xs={s:fetch(s) for s in U}; xs={s:d for s,d in xs.items() if d is not None}
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v.date); v=v.set_index('date').close.astype(float)
def run(k):
 vg=(v>v.rolling(20).mean()*k).shift(1); rows=[]
 for s,d in xs.items():
  d=d.copy(); d.date=pd.to_datetime(d.date); c=d.close.astype(float); vol=np.log(c/c.shift(1)).rolling(20).std(); f=(np.log(c/c.shift(10))/vol).shift(1); r=c.shift(-10)/c-1
  rows.append(pd.DataFrame({'date':d.date,'f':f,'r':r}).dropna())
 x=pd.concat(rows); out=[]
 for dt,g in x.groupby('date'):
  gate=vg.get(dt,np.nan); z=g.f-g.f.median()
  if len(g)>=8 and z.nunique()>1 and g.r.nunique()>1 and pd.notna(gate): out.append((-z if gate else z).corr(g.r,method='spearman'))
 q=pd.Series(out).dropna(); print('k=%.3f dates=%d IC=%.6f ICIR=%.6f hit=%.4f'%(k,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for k in [.90,.95,1.0,1.05,1.10,1.15]: run(k)
