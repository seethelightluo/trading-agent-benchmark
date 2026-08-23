import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d)>40:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Cross-sectional relative momentum: 20d return minus contemporaneous cross-sectional median, smoothed with 5d return; lag one day.
m20=p.pct_change(20); med=m20.median(axis=1); rel=(m20.sub(med,axis=0)+0.5*p.pct_change(5).sub(p.pct_change(5).median(axis=1),axis=0)).shift(1)
rows=[]; sig=[]
for dt in rel.index:
 v=rel.loc[dt].dropna()
 if len(v)>=8:
  sig += [{'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(x)} for s,x in v.items()]
  for h in [1,3,5,10]:
   y=p.pct_change(h).shift(-h).loc[dt,v.index].dropna(); c=v.index.intersection(y.index)
   if len(c)>=8: rows.append((dt,h,len(c),v.loc[c].corr(y.loc[c])))
r=pd.DataFrame(rows,columns=['date','h','n','ic']); print('assets',len(px),'dates',p.index.min(),p.index.max(),'signal_rows',len(sig))
for h in [1,3,5,10]:
 x=r[r.h==h].ic; print('H',h,'dates',len(x),'avg_n',round(r[r.h==h].n.mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
o=pd.DataFrame(sig); o.to_csv('scripts/miner_2_20290222_relative_momentum_signal.csv',index=False); print('coverage',round(len(sig)/(len(rel.index)*len(U)),4),'signal_dates',o.date.nunique())
