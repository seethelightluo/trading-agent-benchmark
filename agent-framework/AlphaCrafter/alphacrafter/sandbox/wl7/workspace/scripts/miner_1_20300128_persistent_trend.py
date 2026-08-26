import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
TODAY='2030-01-25'; H=10
u=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 d=None
 try:d=get_index_daily_data(s,days=4000)
 except Exception: pass
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,days=4000)
  except Exception: d=None
 if d is not None and len(d): px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]

def calc(x):
 r=x.pct_change().dropna()
 if len(x)<130 or len(r)<60:return np.nan
 # medium trend, penalize choppy paths and scale risk
 v=r.iloc[-60:].std()
 if v<=0:return np.nan
 trend=x.iloc[-1]/x.iloc[-81]-1
 consistency=abs(r.iloc[-40:].mean())/(r.iloc[-40:].abs().mean()+1e-12)
 return trend/v*consistency

def run(h):
 rows=[]
 for i in range(120,len(P)-h):
  vals=[]; fw=[]
  for s in u:
   if s not in P:continue
   x=P[s].iloc[:i+1].dropna()
   if i+h>=len(P) or pd.isna(P[s].iloc[i+h]):continue
   f=calc(x)
   if np.isfinite(f):vals.append(f);fw.append(P[s].iloc[i+h]/x.iloc[-1]-1)
  if len(vals)>=8:
   c=np.corrcoef(vals,fw)[0,1]
   if np.isfinite(c):rows.append((P.index[i],c,len(vals)))
 return pd.DataFrame(rows,columns=['date','ic','n'])
for h in [5,10,20,40]:
 z=run(h);print('H',h,'dates',len(z),'avg_n',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'hit',(z.ic>0).mean())
z=run(10); z.to_csv('scripts/miner_1_20300128_persistent_trend_ic.csv',index=False)
# signal artifact at each date, including historical observations for audit
rows=[]
for i in range(120,len(P)-10):
 for s in u:
  if s in P:
   x=P[s].iloc[:i+1].dropna(); f=calc(x)
   if np.isfinite(f):rows.append((P.index[i],s,f))
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20300128_persistent_trend_signal.csv',index=False)
print('regimes')
for a,b in [('2020-01-01','2026-07-15'),('2026-07-16','2028-12-31'),('2029-01-01','2030-01-25')]:
 q=z[(z.date>=a)&(z.date<=b)];print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std() if len(q)>1 else np.nan,(q.ic>0).mean() if len(q) else np.nan)
print('coverage',len(P.columns)/15,'turnover',pd.DataFrame(rows,columns=['date','symbol','signal']).pivot(index='date',columns='symbol',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
print('data',P.index.min(),P.index.max(),'instruments',len(P.columns))
