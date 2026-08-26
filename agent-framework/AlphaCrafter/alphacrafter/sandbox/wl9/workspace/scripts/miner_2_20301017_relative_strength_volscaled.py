import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>150:
  q=d[['date','close']].copy(); q.date=pd.to_datetime(q.date); data[s]=q.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(data).sort_index(); r=p.pct_change()
ret60=p.pct_change(60); rel=ret60.sub(ret60.median(axis=1),axis=0)
vol20=r.rolling(20).std()*np.sqrt(252)
f=(rel/vol20).shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h).div(p)-1; vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 a=pd.Series(vals,index=dates).dropna(); ic=a.mean(); ir=ic/a.std(ddof=1)*np.sqrt(len(a)) if len(a)>1 else np.nan
 print(f'H={h} dates={len(a)} avgN={np.mean(ns):.2f} coverage={np.mean(ns)/len(U):.4f} IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(a>0):.4f}')
fr=p.shift(-20).div(p)-1; vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.Series(dict(vals)).sort_index()
for lo,hi,nm in [('2024-01-01','2026-12-31','2024-26'),('2027-01-01','2029-12-31','2027-29'),('2030-01-01','2030-10-16','2030YTD')]:
 q=a.loc[lo:hi]; print(f'regime={nm} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 else np.nan:.6f}')
print(f'turnover_proxy={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f} instruments={len(data)} dates={len(p)} avg_factor_N={f.notna().sum(axis=1).mean():.2f}')
f.index=f.index.strftime('%Y-%m-%d'); f.to_csv('scripts/miner_2_20301017_relative_strength_volscaled_signal.csv',index_label='date')
