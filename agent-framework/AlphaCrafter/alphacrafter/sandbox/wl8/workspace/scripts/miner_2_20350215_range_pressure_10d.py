import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
series={}
for s in U:
    d=get_stock_daily_data(s, days=6000)
    if d is not None and len(d)>100:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        rng=(d.high-d.low).replace(0,np.nan)
        cl=((d.close-d.open)/rng).clip(-1,1)
        ret=d.close.pct_change()
        f=cl.rolling(10,min_periods=8).mean()/(ret.rolling(20,min_periods=15).std()*np.sqrt(10)+1e-12)
        series[s]=pd.DataFrame({'f':f.shift(1),'close':d.close})
rows=[]
for s,x in series.items():
 y=x.close.pct_change(10).shift(-10); z=pd.concat([x.f,y.rename('y')],axis=1).dropna(); z['s']=s; rows.append(z.reset_index())
a=pd.concat(rows,ignore_index=True)
ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ics.append((dt,g.f.corr(g.y),len(g)))
ic=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date')
r=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True); to=(r.diff().abs().mean(axis=1)/2).dropna()
print('assets',len(series),'dates',len(ic),'avgN',ic.n.mean(),'coverage',len(a)/(len(set(a.date))*len(U)),'avgIC',ic.ic.mean(),'ICIR',ic.ic.mean()/ic.ic.std(ddof=1),'hit',(ic.ic>0).mean(),'recent365',ic.tail(365).ic.mean()/ic.tail(365).ic.std(ddof=1),'turnover',to.mean())
for h in [1,5,10,20]:
 rr=[]
 for s,x in series.items():
  yy=x.close.pct_change(h).shift(-h); zz=pd.concat([x.f,yy.rename('y')],axis=1).dropna(); zz['s']=s; rr.append(zz.reset_index())
 q=pd.concat(rr,ignore_index=True); ii=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ii.append(g.f.corr(g.y))
 print('horizon',h,'IC',np.mean(ii),'ICIR',np.mean(ii)/np.std(ii,ddof=1),'n',len(ii))
