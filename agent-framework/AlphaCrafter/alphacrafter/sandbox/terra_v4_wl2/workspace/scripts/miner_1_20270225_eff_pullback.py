import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try: d=get_stock_daily_data(s,days=2500)
 except Exception:
  try: d=get_index_daily_data(s,days=2500)
  except Exception: d=None
 if d is not None and len(d): D[s]=d.sort_values('date').drop_duplicates('date').set_index('date')
px=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); r=px.pct_change(1); net=px.pct_change(20); eff=net.abs()/r.abs().rolling(20).sum(); f=(net*eff)-px.pct_change(5)*0.5
for h in [1,5,10]:
 vals=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1]))
 a=np.array(vals); print('h',h,'dates',len(a),'instruments',len(D),'mean',np.nanmean(a),'ir',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
rows=[(date,s,f.loc[date,s]) for date in f.index for s in f.columns]; pd.DataFrame(rows,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_1_20270225_eff_pullback.csv',index=False)
print('dates',len(px),'coverage',f.notna().mean().mean())
for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
 vals=[]
 for i in range(len(px)-1):
  if not(lo<=px.index[i].year<=hi): continue
  q=pd.concat([f.iloc[i],px.iloc[i+1]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1]))
 print('regime',lo,hi,'dates',len(vals),'ic',np.nanmean(vals) if vals else np.nan)
