import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
TODAY='2030-02-11'; H=10
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 d=None
 try:d=get_index_daily_data(s,days=4000)
 except Exception: pass
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,days=4000)
  except Exception:d=None
 if d is not None and len(d):px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]

def calc(i,s):
 x=P[s].iloc[:i+1].dropna()
 if len(x)<120:return None
 rr=x.pct_change().dropna(); a=rr.iloc[-20:]; b=rr.iloc[-60:]; v=b.std()
 if v<=0 or a.abs().mean()<=0:return None
 # medium trend scaled by volatility and rewarded for directional persistence
 f=(x.iloc[-1]/x.iloc[-61]-1)/v * abs(a.mean())/a.abs().mean()
 if i+H>=len(P) or pd.isna(P[s].iloc[i+H]):return None
 return f,P[s].iloc[i+H]/x.iloc[-1]-1
rows=[]
for i in range(120,len(P)-H):
 for s in u:
  z=calc(i,s)
  if z is not None and np.isfinite(z[0]) and np.isfinite(z[1]):rows.append((P.index[i],s,*z))
df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
ics=df.groupby('date').apply(lambda z:z.factor.corr(z.fwd),include_groups=False).dropna()
r=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('candidate=persistent_trend60 consistency20; cutoff',TODAY,'dates',len(ics),'avg_names',df.groupby('date').size().mean(),'coverage',df.symbol.nunique()/len(u),'IC10',ics.mean(),'ICIR10',ics.mean()/ics.std(),'hit10',(ics>0).mean(),'turnover',r.diff().abs().mean(axis=1).dropna().mean())
for h in [1,5,10,20,40]:
 aa=[]
 for i in range(120,len(P)-h):
  z=[]
  for s in u:
   x=P[s].iloc[:i+1].dropna()
   if len(x)<120 or i+h>=len(P) or pd.isna(P[s].iloc[i+h]):continue
   rr=x.pct_change().dropna();a=rr.iloc[-20:];b=rr.iloc[-60:];v=b.std()
   if v>0 and a.abs().mean()>0:
    f=(x.iloc[-1]/x.iloc[-61]-1)/v*abs(a.mean())/a.abs().mean(); z.append((f,P[s].iloc[i+h]/x.iloc[-1]-1))
  if len(z)>=8:aa.append(pd.Series([q[0] for q in z]).corr(pd.Series([q[1] for q in z])))
 print('horizon',h,'IC',np.nanmean(aa),'ICIR',np.nanmean(aa)/np.nanstd(aa),'dates',len(aa))
# regime slices
for label,lo,hi in [('early','2020-01-01','2024-12-31'),('mid','2025-01-01','2027-12-31'),('late','2028-01-01',TODAY)]:
 q=ics.loc[(ics.index>=lo)&(ics.index<=hi)];print('regime',label,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std())
df.to_csv('scripts/miner_3_20300211_persistent_trend_signal.csv',index=False)
