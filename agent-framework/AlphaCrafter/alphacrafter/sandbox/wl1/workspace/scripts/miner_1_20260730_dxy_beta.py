import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
def load(s):
    try: d=get_index_daily_data(s,3000)
    except Exception: d=None
    if d is None:
        try: d=get_stock_daily_data(s,3000)
        except Exception: d=None
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']).dt.normalize(); return d.set_index('date')
for s in U: 
    d=load(s)
    if d is not None: F[s]=d
try: macro=load('DXY')
except Exception: macro=None
m=macro.close.astype(float).pct_change()
rows=[]
for s,d in F.items():
    r=d.close.astype(float).pct_change(); z=pd.concat([r,m],axis=1,keys=['r','m']).sort_index()
    # DXY beta: negative exposure to dollar, estimated on completed data
    cov=z.r.rolling(60,min_periods=40).cov(z.m); var=z.m.rolling(60,min_periods=40).var()
    f=-cov/var
    q=pd.DataFrame({'f':f,'fr':r.shift(-1),'s':s}).loc[:'2026-07-15'].reset_index(); rows.append(q)
x=pd.concat(rows,ignore_index=True).dropna(subset=['f','fr'])
ics=x.groupby('date').apply(lambda q:q.f.corr(q.fr) if len(q)>=8 else np.nan).dropna()
print('daily',len(ics),round(ics.mean(),6),round(ics.mean()/ics.std(ddof=1),6),round((ics>0).mean(),4),'meanN',round(x.groupby('date').size().mean(),2),'coverage',round(len(x)/(len(F)*len(pd.date_range('2020-01-01','2026-07-15'))),4))
for h in [5,10]:
 x['fh']=np.nan
 for s in x.s.unique():
  ix=x.s==s; c=F[s].close.astype(float); x.loc[ix,'fh']=x.loc[ix,'date'].map(c.shift(-h)/c-1)
 a=x.dropna(subset=['fh']).groupby('date').apply(lambda q:q.f.corr(q.fh) if len(q)>=8 else np.nan).dropna(); print(f'{h}d',len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6))
print('regimes',x.assign(y=x.date.dt.year).groupby('y').apply(lambda q:q.groupby('date').apply(lambda z:z.f.corr(z.fr) if len(z)>=8 else np.nan).mean()).round(5).to_dict())
print('dates',x.date.min(),x.date.max(),'names',x.s.nunique())
