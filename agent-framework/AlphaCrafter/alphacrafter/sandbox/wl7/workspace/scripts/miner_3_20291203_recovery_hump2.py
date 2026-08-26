import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

TODAY='2029-12-03'; H=20
acct=get_account_dict(); u=acct.get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: moderate drawdown recovery, but use 60d peak and 20d recovery slope; rewards recovery from ~8-18% drawdown.
px={}
for s in u:
 
 try: d=get_index_daily_data(s, days=4000)
 except Exception: d=None
 if d is None or len(d)<150:
  try: d=get_stock_daily_data(s, days=4000)
  except Exception: d=None
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); P=P.loc[:TODAY]
rows=[]
for i in range(100,len(P)-H):
 date=P.index[i]; vals={}; fw={}
 for s in u:
  x=P[s].iloc[:i+1].dropna()
  if len(x)<100 or i+H>=len(P) or pd.isna(P[s].iloc[i+H]): continue
  peak=x.iloc[:-1].rolling(60).max().iloc[-1] if len(x)>=61 else np.nan
  dd=x.iloc[-1]/peak-1 if peak and np.isfinite(peak) else np.nan
  r20=x.iloc[-1]/x.iloc[-21]-1 if len(x)>=21 else np.nan
  vol=x.pct_change().iloc[-21:-1].std()*np.sqrt(20) if len(x)>=22 else np.nan
  if np.isfinite(dd) and np.isfinite(r20) and np.isfinite(vol) and vol>0:
   # smooth hump centered at -10%, plus positive recovery velocity
   vals[s]=(dd+0.10)*np.exp(-((dd+0.10)/0.11)**2) + 0.20*r20/vol
   fw[s]=P[s].iloc[i+H]/x.iloc[-1]-1
 for s in vals: rows.append((date,s,vals[s],fw[s]))
df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna()
ics=df.groupby('date').apply(lambda z: z.factor.corr(z.fwd),include_groups=False).dropna()
# rank turnover based on daily rank vectors on common dates
r=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
turn=r.diff().abs().mean(axis=1).dropna().mean()
print('dates',len(ics),'avg_names',df.groupby('date').size().mean(),'coverage',df.symbol.nunique()/len(u),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(),'hit',(ics>0).mean(),'turnover',turn)
for a,b in [(ics.index[0],ics.index[int(len(ics)*.5)]),(ics.index[int(len(ics)*.5)],ics.index[-1])]:
 q=ics.loc[(ics.index>=a)&(ics.index<=b)]; print('regime',a,b,len(q),q.mean(),q.mean()/q.std())
for h in [5,10,20,40]:
 rr=[]
 for i in range(100,len(P)-h):
  z=[]
  for s in u:
   x=P[s].iloc[:i+1].dropna()
   if len(x)<100 or i+h>=len(P) or pd.isna(P[s].iloc[i+h]): continue
   peak=x.iloc[:-1].rolling(60).max().iloc[-1]; dd=x.iloc[-1]/peak-1
   r20=x.iloc[-1]/x.iloc[-21]-1; vol=x.pct_change().iloc[-21:-1].std()*np.sqrt(20)
   if np.isfinite(dd) and vol>0: z.append((s,(dd+.10)*np.exp(-((dd+.10)/.11)**2)+.2*r20/vol,P[s].iloc[i+h]/x.iloc[-1]-1))
  if len(z)>=8: rr.append(pd.Series([q[1] for q in z]).corr(pd.Series([q[2] for q in z])))
 print('horizon',h,'IC',np.nanmean(rr),'dates',len(rr))
df.to_csv('scripts/miner_3_20291203_recovery_hump2_signal.csv',index=False)
