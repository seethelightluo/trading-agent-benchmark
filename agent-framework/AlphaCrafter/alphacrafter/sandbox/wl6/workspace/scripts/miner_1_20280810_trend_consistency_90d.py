import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
from scipy.stats import spearmanr
acct=get_account_dict(); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in syms:
    d=None
    for fn in (get_index_daily_data,get_stock_daily_data):
      try: d=fn(s,3000)
      except Exception: pass
      if d is not None: break
    if d is not None and len(d)>120:
      d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
      frames[s]=d['close'].astype(float)
px=pd.DataFrame(frames).sort_index(); rets=px.pct_change()
sig=rets.gt(0).rolling(90,min_periods=75).mean().shift(1)-.5
ics=[]; rows=[]
for dt in sig.index:
 x=sig.loc[dt]; y=(px.shift(-1)/px-1).loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): ics.append(ic); rows.append((dt,ic,len(z)))
a=np.array(ics); rank=sig.rank(axis=1,pct=True)
print('symbols',len(frames),'dates',len(rows),'avg_n',np.mean([r[2] for r in rows]),'coverage',sig.notna().sum().sum()/(len(sig)*len(syms)))
print('daily_ic %.8f icir %.8f hit %.5f turnover %.8f' %(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.nanmean(rank.diff().abs().mean(axis=1))))
for name,lo,hi in [('2020-22',2020,2022),('2023-24',2023,2024),('2025-26',2025,2026),('2027-28',2027,2028)]:
 q=np.array([r[1] for r in rows if lo<=r[0].year<=hi]); print(name,len(q),('%.8f'%q.mean()) if len(q) else 'NA')
for h in [5,10]:
 ff=px.shift(-h)/px-1; aa=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'n',len(aa),'ic',np.nanmean(aa))
