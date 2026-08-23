import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=None
    for fn in (get_index_daily_data,get_stock_daily_data):
      try: x=fn(s,days=3000); break
      except (FileNotFoundError,KeyError): pass
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
print('loaded',len(D),sorted(D))
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
mom=p.pct_change(20); rel=mom.sub(mom.median(axis=1),axis=0)
vol=r.rolling(40).std()*np.sqrt(20); shock=r.rolling(5).sum().abs()
f=rel.div(vol.replace(0,np.nan))/(1+shock); fr=p.shift(-10).div(p)-1
rows=[]; dates=[]; ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(d); ns.append(len(z))
ic=np.array(rows); print('dates',len(ic),'avg_n',np.mean(ns),'coverage %.4f'%(len(dates)/len(f)))
print('IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(ic),np.nanmean(ic)/np.nanstd(ic,ddof=1),np.mean(ic>0)))
for label,lo,hi in [('2020-22','2020','2023'),('2023-24','2023','2025'),('2025-26','2025','2027'),('2027-28','2027','2029'),('2029','2029','2030')]:
 q=ic[(np.array(dates)>=pd.Timestamp(lo))&(np.array(dates)<pd.Timestamp(hi))]; print(label,len(q), '%.6f'%np.nanmean(q) if len(q) else 'NA')
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
