import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=None
 for fn in (get_stock_daily_data,get_index_daily_data):
  try: d=fn(s,days=6000)
  except (FileNotFoundError,KeyError): pass
  if d is not None and len(d): break
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
print('loaded',len(px),sorted(px))
p=pd.DataFrame(px).sort_index()
r5=p.pct_change(5); r60=p.pct_change(60); sig=(r60-r5).shift(1); fwd=p.shift(-10)/p-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.sum()/(len(q)*15)); print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 z=q.loc[a:b]; print('regime',a,b,len(z),z.ic.mean(),z.ic.mean()/z.ic.std())
for h in [5,10,20,40]:
 yy=p.shift(-h)/p-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals))
out=sig.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20350316_trend_pullback_signal.csv'); q.to_csv('scripts/miner_1_20350316_trend_pullback_ic.csv')
