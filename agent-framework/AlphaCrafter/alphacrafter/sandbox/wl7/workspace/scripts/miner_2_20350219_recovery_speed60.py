import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Recovery speed: rebound from trailing 60d trough per day elapsed since trough.
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
D=pd.DataFrame(px).sort_index(); D=D.loc[:pd.Timestamp('2035-02-17')]
rows=[]
for i in range(60,len(D)-20):
 dt=D.index[i]; vals=[]; fwd=[]
 for s in U:
  x=D[s].iloc[:i+1]
  if len(x)<60 or pd.isna(x.iloc[-1]): continue
  w=x.iloc[-60:]; trough=w.idxmin(); age=max(0,(w.index[-1]-trough).days)
  # observations since trough, not calendar days, bounded
  n=max(1,len(w.loc[trough:])-1)
  fac=(w.iloc[-1]/w.min()-1.0)/n
  r=D[s].iloc[i+20]/D[s].iloc[i]-1 if pd.notna(D[s].iloc[i+20]) else np.nan
  if np.isfinite(fac) and np.isfinite(r): vals.append(fac);fwd.append(r)
 if len(vals)>=8:
  rows.append((dt,spearmanr(vals,fwd).statistic,len(vals)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit', (r.ic>0).mean())
for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034'),('2035','2035')]:
 q=r.loc[a:b].ic; print(a,b,'n',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('last',r.tail(10).to_string())
# rank turnover proxy
F=[]
for i in range(60,len(D)-20):
 vals=[]
 for s in U:
  x=D[s].iloc[:i+1];w=x.iloc[-60:]
  if len(w)<60 or w.isna().any(): vals.append(np.nan);continue
  trough=w.idxmin();n=max(1,len(w.loc[trough:])-1); vals.append((w.iloc[-1]/w.min()-1)/n)
 F.append(vals)
fr=pd.DataFrame(F,columns=U).rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print('rank_turnover_proxy',fr)
# save signal aligned dates
out=[]
for i in range(60,len(D)-20):
 for s in U:
  x=D[s].iloc[:i+1];w=x.iloc[-60:]
  if len(w)<60 or w.isna().any(): continue
  tr=w.idxmin();n=max(1,len(w.loc[tr:])-1);out.append({'date':D.index[i],'symbol':s,'signal':(w.iloc[-1]/w.min()-1)/n})
pd.DataFrame(out).to_csv('scripts/miner_2_20350219_recovery_speed60_signal.csv',index=False)
