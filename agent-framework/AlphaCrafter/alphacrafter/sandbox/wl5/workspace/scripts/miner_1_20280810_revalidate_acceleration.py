import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();px[s]=d.set_index('date').close
p=pd.DataFrame(px).sort_index().ffill();r=p.pct_change();ret5=r.rolling(5).sum(); prior=(r.rolling(20).sum()-ret5)/3;vol=r.rolling(20).std()*np.sqrt(20);f=-(ret5-prior)/(vol+1e-8)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(y[ok]),ok.mean()))
d=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna(); z=d[d.date>='2027-01-01']
for label,q in [('2027+',z),('last250',d.tail(250)),('last500',d.tail(500))]: print(label,'dates',len(q),'names',round(q.coverage.mean()*15,2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
