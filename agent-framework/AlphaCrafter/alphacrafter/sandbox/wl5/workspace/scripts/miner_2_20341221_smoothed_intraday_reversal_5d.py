import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def sp(a,b): return pd.Series(a).rank().corr(pd.Series(b).rank())
rows=[]; sig=[]; mats={}
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<300: continue
 d=d.sort_values('date'); r=d['close']/d['open']-1; v=d['pct_change'].rolling(20,min_periods=15).std()
 f=-(r.rolling(5,min_periods=5).mean()/v).clip(-6,6); fr=d['close'].shift(-10)/d['close']-1
 for i in range(len(d)-10):
  if pd.notna(f.iloc[i]) and pd.notna(fr.iloc[i]): rows.append((d['date'].iloc[i],s,f.iloc[i],fr.iloc[i])); sig.append((d['date'].iloc[i],s,f.iloc[i]))
 mats[s]=pd.Series(f.values,index=d['date'])
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
groups=x.groupby('date'); q=[]; ns=[]
for dt,g in groups:
 if len(g)>=8: q.append(sp(g.factor,g.fwd)); ns.append(len(g))
q=np.array(q); print('dates',len(q),'meanN',np.mean(ns),'coverage',len(x)/(len(q)*15)); print('IC10',q.mean(),'ICIR_daily',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',np.mean(q>0))
for h in [5,20]:
 z=[]
 for s in U:
  d=get_stock_daily_data(s,2800)
  if d is None: continue
  d=d.sort_values('date'); r=d['close']/d['open']-1; v=d['pct_change'].rolling(20,min_periods=15).std(); f=-(r.rolling(5,min_periods=5).mean()/v).clip(-6,6); fr=d['close'].shift(-h)/d['close']-1
  for i in range(len(d)-h):
   if pd.notna(f.iloc[i]) and pd.notna(fr.iloc[i]): z.append((d['date'].iloc[i],f.iloc[i],fr.iloc[i]))
 z=pd.DataFrame(z,columns=['date','f','r']); a=[sp(g.f,g.r) for _,g in z.groupby('date') if len(g)>=8]; print('decay',h,np.nanmean(a),len(a))
# rank turnover over adjacent dates, averaged over available names
wide=pd.DataFrame({s:v for s,v in mats.items()}).sort_index(); ranks=wide.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean())
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20341221_smoothed_intraday_reversal_5d_signal.csv',index=False)
