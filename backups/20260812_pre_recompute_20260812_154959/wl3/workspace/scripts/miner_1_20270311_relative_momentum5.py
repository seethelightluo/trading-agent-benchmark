import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames=[]
for s in U:
    d=get_stock_daily_data(s,2600)
    if d is None or len(d)<100: d=get_index_daily_data(s,2600)
    if d is not None and len(d): frames.append(d[['date','close']].assign(symbol=s))
px=pd.concat(frames).pivot(index='date',columns='symbol',values='close').sort_index()
r=px.pct_change()
# Cross-asset relative short-term momentum: each asset's 5-day return
# demeaned by the contemporaneous cross-sectional median, reducing common beta.
raw=px.pct_change(5)
f=raw.sub(raw.median(axis=1),axis=0)
cut=pd.Timestamp('2027-03-10'); f=f.loc[:cut]
forward=px.pct_change().shift(-1).loc[:cut]
q=[]; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],forward.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic):
   q.append((dt,ic,len(z)))
   rows.extend((dt,s,float(f.loc[dt,s])) for s in z.index)
d=pd.DataFrame(q,columns=['date','ic','n'])
print('assets',len(px.columns),'dates',len(d),'avg_n',d.n.mean(),'coverage',len(rows)/(len(f.index)*len(U)))
print('IC %.6f ICIR %.6f hit %.4f'%(d.ic.mean(),d.ic.mean()/d.ic.std(ddof=1),(d.ic>0).mean()))
for a,b in [('2020','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-10')]:
 x=d[(d.date>=a)&(d.date<=b)]; print('REG',a,b,'dates',len(x),'IC %.6f ICIR %.6f'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1)) if len(x)>1 else 'insufficient')
for h in [3,5,10]:
 fy=px.pct_change(h).shift(-h).loc[:cut]; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('DECAY',h,'IC %.6f ICIR %.6f dates %d'%(np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals,ddof=1),len(vals)))
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20270311_relative_momentum5_signal.csv',index=False)
