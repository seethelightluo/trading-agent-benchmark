import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; frames={}
for s in U:
 try:d=get_index_daily_data(s,3000)
 except Exception:d=None
 if d is None:
  try:d=get_stock_daily_data(s,3000)
  except Exception:d=None
 if d is not None and len(d): frames[s]=d.assign(date=pd.to_datetime(d.date).dt.normalize()).set_index('date')
rows=[]
for s,d in frames.items():
 c=pd.to_numeric(d.close,errors='coerce');v=pd.to_numeric(d.volume,errors='coerce');r=c.pct_change();vs=np.log(v.replace(0,np.nan)/v.rolling(20,min_periods=15).median());f=vs*np.sign(r.rolling(5,min_periods=5).sum());rows.append(pd.DataFrame({'f':f,'fr':r.shift(-1),'s':s}).loc[:'2026-07-15'].reset_index())
x=pd.concat(rows,ignore_index=True).dropna(subset=['f','fr']); a=x.groupby('date').apply(lambda q:q.f.corr(q.fr) if len(q)>=8 else np.nan).dropna();print('daily',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'meanN',round(x.groupby('date').size().mean(),2),'names',x.s.nunique(),'dates',x.date.min(),x.date.max())
for h in [5,10]:
 vals=[]
 for s,d in frames.items():
  c=pd.to_numeric(d.close,errors='coerce');vs=np.log(pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)/pd.to_numeric(d.volume,errors='coerce').rolling(20,min_periods=15).median());f=vs*np.sign(c.pct_change().rolling(5,min_periods=5).sum());z=pd.DataFrame({'f':f,'y':c.shift(-h)/c-1}).loc[:'2026-07-15'];z['s']=s;vals.append(z)
 q=pd.concat(vals).dropna(); aa=q.groupby('date').apply(lambda z:z.f.corr(z.y) if len(z)>=8 else np.nan).dropna();print('decay',h,len(aa),round(aa.mean(),6),round(aa.mean()/aa.std(ddof=1),6))
print('regimes',x.assign(y=x.date.dt.year).groupby('y').apply(lambda q:q.groupby('date').apply(lambda z:z.f.corr(z.fr) if len(z)>=8 else np.nan).mean()).round(5).to_dict())
