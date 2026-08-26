import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  q=d[['date','close']].copy(); q.date=pd.to_datetime(q.date); data[s]=q.drop_duplicates('date').set_index('date').sort_index()
p=pd.DataFrame({s:x.close.astype(float) for s,x in data.items()}).sort_index(); r=p.pct_change()
direction=p.pct_change(60); path=r.abs().rolling(60,min_periods=45).sum(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
# Fade smooth, efficient 60d trends: negative trend efficiency, scaled by recent risk; lag one day.
f=(-direction/(path+1e-12)/(vol+1e-8)).shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h).div(p)-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=pd.Series(vals).dropna(); ic=a.mean(); ir=ic/a.std(ddof=1)*np.sqrt(len(a))
 print(f'H={h} dates={len(a)} avgN={np.mean(ns):.2f} coverage={np.mean(ns)/len(U):.4f} IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(a>0):.4f}')
print(f'turnover_proxy={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f} instruments={len(data)} dates={len(p)} avg_factor_N={f.notna().sum(axis=1).mean():.2f}')
f.index=f.index.strftime('%Y-%m-%d'); f.to_csv('scripts/miner_2_20301212_efficiency_reversal_signal.csv',index_label='date')
