import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    try: d=get_stock_daily_data(s, days=6000)
    except Exception as e: print('missing',s,e); continue
    if d is not None and len(d)>150:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index(); frames[s]=d['close'].astype(float)
px=pd.DataFrame(frames).sort_index()
ret20=px/px.shift(20)-1; high=px.rolling(120,min_periods=60).max(); dd=(px/high-1).abs()
signal=(ret20.rolling(5,min_periods=5).mean()/(0.01+dd.rolling(40,min_periods=20).mean())).shift(1)
fwd=px.shift(-10)/px-1; ics=[]; rows=[]
for dt in signal.index:
 z=pd.concat([signal.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(ic): ics.append(ic); rows.append((dt,ic,len(z)))
ics=np.array(ics); print('cutoff',px.index.max().date(),'dates',len(ics),'assets',len(frames),'avg_n',np.mean([r[2] for r in rows])); print('IC %.8f ICIR %.8f hit %.4f'%(ics.mean(),ics.mean()/(ics.std(ddof=1)+1e-12),np.mean(ics>0)))
ranks=signal.rank(axis=1,pct=True); turns=[]
for a,b in zip(ranks.index[:-1],ranks.index[1:]):
 z=pd.concat([ranks.loc[a],ranks.loc[b]],axis=1).dropna()
 if len(z)>=8: turns.append(np.mean(abs(z.iloc[:,0]-z.iloc[:,1])))
print('turnover',np.mean(turns),'coverage',np.mean([r[2] for r in rows])/len(frames))
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; aa=[]
 for dt in signal.index:
  z=pd.concat([signal.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(aa))
for label,lo,hi in [('2020-24','2020','2024-12-31'),('2025-30','2025','2030-12-31'),('2031-35','2031','2035-12-31'),('recent120',str(px.index.max()-pd.Timedelta(days=180)),str(px.index.max()))]:
 a=[v for d,v,n in rows if str(d.date())>=lo and str(d.date())<=hi]; print(label,len(a),np.mean(a) if a else np.nan,(np.mean(a)/(np.std(a,ddof=1)+1e-12)) if len(a)>1 else np.nan)
signal.to_csv('scripts/miner_2_20351026_smoothed_recovery_signal.csv')
