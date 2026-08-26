import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def rc(a,b): return pd.Series(a).rank().corr(pd.Series(b).rank())
rows=[]; sig=[]; mats={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<300: continue
 d=d.sort_values('date').reset_index(drop=True)
 gap=d.open/d.close.shift(1)-1
 vol=d['pct_change'].rolling(20,min_periods=15).std()
 f=(-(gap.rolling(3,min_periods=3).mean()/vol).clip(-6,6))
 fr=d.close.shift(-10)/d.close-1
 for i in range(len(d)-10):
  if pd.notna(f.iloc[i]) and pd.notna(fr.iloc[i]):
   rows.append((d.date.iloc[i],s,f.iloc[i],fr.iloc[i])); sig.append((d.date.iloc[i],s,f.iloc[i]))
 mats[s]=pd.Series(f.values,index=d.date)
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']); q=[]; ns=[]
for dt,g in x.groupby('date'):
 if len(g)>=8: q.append(rc(g.factor,g.fwd)); ns.append(len(g))
q=np.array(q)
print('dates',len(q),'meanN',np.mean(ns),'coverage',len(x)/(len(q)*15))
print('IC10',np.nanmean(q),'ICIR_daily',np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(252),'hit',np.mean(q>0))
for h in [5,20]:
 z=[]
 for s in U:
  d=get_stock_daily_data(s,3000)
  if d is None: continue
  d=d.sort_values('date').reset_index(drop=True); gap=d.open/d.close.shift(1)-1; vol=d['pct_change'].rolling(20,min_periods=15).std(); f=(-(gap.rolling(3,min_periods=3).mean()/vol).clip(-6,6)); fr=d.close.shift(-h)/d.close-1
  for i in range(len(d)-h):
   if pd.notna(f.iloc[i]) and pd.notna(fr.iloc[i]): z.append((d.date.iloc[i],f.iloc[i],fr.iloc[i]))
 z=pd.DataFrame(z,columns=['date','f','r']); a=[rc(g.f,g.r) for _,g in z.groupby('date') if len(g)>=8]
 print('decay',h,np.nanmean(a),len(a))
wide=pd.DataFrame({s:v for s,v in mats.items()}).sort_index(); ranks=wide.rank(axis=1,pct=True)
print('turnover',ranks.diff().abs().mean(axis=1).mean())
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20350118_gap_reversal_3d_signal.csv',index=False)
