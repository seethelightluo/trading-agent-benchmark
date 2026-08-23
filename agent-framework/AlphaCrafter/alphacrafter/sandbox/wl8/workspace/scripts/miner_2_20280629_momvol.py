import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=4000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change()
mom=px.shift(1)/px.shift(21)-1
vol=r.shift(1).rolling(20,min_periods=15).std()*np.sqrt(20)
f=mom/vol
out=[]
for dt in px.index:
    row=f.loc[dt]; j=px.index.get_loc(dt)
    for h in [1,3,5,10]:
        if j+h>=len(px): continue
        fr=px.iloc[j+h]/px.iloc[j]-1
        z=pd.concat([row,fr],axis=1).dropna()
        if len(z)>=8: out.append((dt,h,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
o=pd.DataFrame(out,columns=['date','h','ic','n'])
print('rows/dates/instruments',len(o),o.date.nunique(),px.shape[1])
for h in [1,3,5,10]:
 z=o[o.h==h].dropna(); ic=z.ic
 print(h,'dates',len(z),'avgN',z.n.mean(),'IC %.6f ICIR %.6f hit %.3f'% (ic.mean(),ic.mean()/ic.std(),(ic>0).mean()))
 for lab,mask in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027),('2028',z.date.dt.year==2028),('recent180',z.date>=z.date.max()-pd.Timedelta(days=280))]:
  q=z[mask].ic
  if len(q): print(' ',lab,len(q),'%.6f %.6f'%(q.mean(),q.mean()/q.std()))
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_2_20280629_momvol_signal.csv',index=False)
