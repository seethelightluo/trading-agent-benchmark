import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
TODAY='2030-01-14'; H=10
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 d=None
 try:d=get_index_daily_data(s,days=4000)
 except:pass
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,days=4000)
  except: d=None
 if d is not None and len(d): px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]
rows=[]
for i in range(120,len(P)-H):
 date=P.index[i]
 for s in u:
  x=P[s].iloc[:i+1].dropna()
  if len(x)<120 or i+H>=len(P) or pd.isna(P[s].iloc[i+H]): continue
  r=x.pct_change().dropna()
  if len(r)<60: continue
  r20=r.iloc[-20:]; r60=r.iloc[-60:]
  vol=r60.std()
  if vol<=0: continue
  # Persistent trend: medium momentum, rewarded only when daily direction is consistent.
  f=(x.iloc[-1]/x.iloc[-61]-1)/vol * abs(r20.mean())/r20.abs().mean()
  fw=P[s].iloc[i+H]/x.iloc[-1]-1
  if np.isfinite(f) and np.isfinite(fw): rows.append((date,s,f,fw))
df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna()
ics=df.groupby('date').apply(lambda z:z.factor.corr(z.fwd),include_groups=False).dropna()
r=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('dates',len(ics),'avg_names',df.groupby('date').size().mean(),'coverage',df.symbol.nunique()/len(u),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(),'hit',(ics>0).mean(),'turnover',r.diff().abs().mean(axis=1).dropna().mean())
for h in [5,10,20,40]:
 aa=[]
 for i in range(120,len(P)-h):
  z=[]
  for s in u:
   x=P[s].iloc[:i+1].dropna()
   if len(x)<120 or i+h>=len(P) or pd.isna(P[s].iloc[i+h]):continue
   rr=x.pct_change().dropna(); r20=rr.iloc[-20:]; r60=rr.iloc[-60:]; v=r60.std()
   if v>0 and r20.abs().mean()>0:z.append(((x.iloc[-1]/x.iloc[-61]-1)/v*abs(r20.mean())/r20.abs().mean(),P[s].iloc[i+h]/x.iloc[-1]-1))
  if len(z)>=8: aa.append(pd.Series([q[0] for q in z]).corr(pd.Series([q[1] for q in z])))
 print('horizon',h,'IC',np.nanmean(aa),'dates',len(aa))
df.to_csv('scripts/miner_3_20300114_persistent_trend_signal.csv',index=False)
