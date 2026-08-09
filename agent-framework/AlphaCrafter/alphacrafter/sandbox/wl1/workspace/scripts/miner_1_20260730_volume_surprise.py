import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    try: d=get_index_daily_data(s,3000)
    except Exception: d=None
    if d is None:
        try: d=get_stock_daily_data(s,3000)
        except Exception: d=None
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']).dt.normalize(); frames[s]=d.set_index('date')
rows=[]
for s,d in frames.items():
    if 'volume' not in d or 'close' not in d: continue
    close=pd.to_numeric(d.close,errors='coerce'); vol=pd.to_numeric(d.volume,errors='coerce')
    r=close.pct_change(); vs=np.log(vol.replace(0,np.nan)/vol.rolling(20,min_periods=15).median())
    f=vs * np.sign(r.rolling(5,min_periods=5).sum())
    z=pd.DataFrame({'f':f,'fr':r.shift(-1)}).loc[:'2026-07-15']; z['s']=s; rows.append(z.reset_index())
x=pd.concat(rows,ignore_index=True).dropna(subset=['f','fr'])
for h in [5,10]:
 x['fh']=np.nan
 for s in x.s.unique():
  ix=x.s==s; c=pd.to_numeric(frames[s].close,errors='coerce'); x.loc[ix,'fh']=x.loc[ix,'date'].map(c.shift(-h)/c-1)
 q=x.dropna(subset=['f','fh']); a=q.groupby('date').apply(lambda z:z.f.corr(z.fh) if len(z)>=8 else np.nan).dropna(); print(f'{h}d',len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4))
ics=x.groupby('date').apply(lambda q: q.f.corr(q.fr) if len(q)>=8 else np.nan).dropna()
print('daily',len(ics),round(ics.mean(),6),round(ics.mean()/ics.std(ddof=1),6),round((ics>0).mean(),4),'meanN',round(x.groupby('date').size().mean(),2),'coverage',round(len(x)/(len(frames)*len(pd.date_range('2020-01-01','2026-07-15'))),4))
print('regimes',x.assign(y=x.date.dt.year).groupby('y').apply(lambda q:q.groupby('date').apply(lambda z:z.f.corr(z.fr) if len(z)>=8 else np.nan).mean()).round(5).to_dict())
print('dates',x.date.min(),x.date.max(),'names',x.s.nunique())
