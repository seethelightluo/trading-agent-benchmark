import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ic(a,b): return pd.Series(a).rank().corr(pd.Series(b).rank())
rows=[]; sig=[]; mats={}
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<300: continue
 d=d.sort_values('date').reset_index(drop=True); intr=d['close']/d['open']-1; vol=d['pct_change'].rolling(40,min_periods=25).std(); f=(intr.rolling(10,min_periods=10).sum()/vol).clip(-8,8); fr=d['close'].shift(-10)/d['close']-1
 for i in range(len(d)-10):
  if pd.notna(f.iloc[i]) and pd.notna(fr.iloc[i]): rows.append((d.date.iloc[i],s,float(f.iloc[i]),float(fr.iloc[i]))); sig.append((d.date.iloc[i],s,float(f.iloc[i])))
 mats[s]=pd.Series(f.values,index=d.date)
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']); ics=[]; ns=[]
for _,g in x.groupby('date'):
 if len(g)>=8:
  v=ic(g.factor,g.fwd)
  if pd.notna(v): ics.append(v); ns.append(len(g))
a=np.array(ics); ir=float(a.mean()/a.std(ddof=1)*np.sqrt(252))
print('dates',len(a),'meanN',round(float(np.mean(ns)),3),'coverage',round(len(x)/(len(a)*15),4)); print('IC10',round(float(a.mean()),6),'ICIR_daily',round(ir,6),'hit',round(float((a>0).mean()),4))
for h in [5,20]:
 z=[]
 for s in U:
  d=get_stock_daily_data(s,2800)
  if d is None: continue
  d=d.sort_values('date').reset_index(drop=True); intr=d['close']/d['open']-1; vol=d['pct_change'].rolling(40,min_periods=25).std(); f=(intr.rolling(10,min_periods=10).sum()/vol).clip(-8,8); fr=d['close'].shift(-h)/d['close']-1
  for i in range(len(d)-h):
   if pd.notna(f.iloc[i]) and pd.notna(fr.iloc[i]): z.append((d.date.iloc[i],f.iloc[i],fr.iloc[i]))
 z=pd.DataFrame(z,columns=['date','f','r']); aa=[ic(g.f,g.r) for _,g in z.groupby('date') if len(g)>=8]; print('decay',h,'IC',round(float(np.nanmean(aa)),6),'dates',len(aa))
wide=pd.DataFrame(mats).sort_index(); ranks=wide.rank(axis=1,pct=True); print('turnover',round(float(ranks.diff().abs().mean(axis=1).mean()),6)); pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20350201_intraday_momentum_10d_signal.csv',index=False)
