import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
TODAY='2030-03-11'; H=10
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 d=None
 try:d=get_index_daily_data(s,days=4000)
 except Exception: pass
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,days=4000)
  except Exception:d=None
 if d is not None and len(d): px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]

def vals(i,h):
 out=[]
 for s in u:
  if s not in P: continue
  x=P[s].iloc[:i+1].dropna()
  if len(x)<100 or i+h>=len(P) or pd.isna(P[s].iloc[i+h]): continue
  lr=np.log(x).diff(); vol=lr.iloc[-61:-1].std()
  if not np.isfinite(vol) or vol<=0: continue
  # medium trend strength, penalized when recent path is choppy
  mom=x.iloc[-1]/x.iloc[-61]-1
  consistency=(lr.iloc[-40:-1]>0).mean()-(lr.iloc[-40:-1]<0).mean()
  f=mom/vol*(0.5+0.5*consistency)
  out.append((s,f,P[s].iloc[i+h]/x.iloc[-1]-1))
 return out
rows=[]
for i in range(100,len(P)-H):
 z=vals(i,H)
 if len(z)>=8:
  med=np.median([q[1] for q in z])
  rows += [(P.index[i],s,f-med,fw) for s,f,fw in z]
df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna()
ic=df.groupby('date').apply(lambda z:z.factor.corr(z.fwd),include_groups=False).dropna()
r=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('universe',len(u),'available',len(px),'dates',len(ic),'avg_names',df.groupby('date').size().mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'turnover',r.diff().abs().mean(axis=1).dropna().mean())
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030')]:
 q=ic.loc[a:b]; print('regime',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
for h in [1,5,10,20,40]:
 aa=[]
 for i in range(100,len(P)-h):
  z=vals(i,h)
  if len(z)>=8:
   med=np.median([q[1] for q in z]); aa.append(pd.Series([q[1]-med for q in z]).corr(pd.Series([q[2] for q in z])))
 print('horizon',h,'dates',len(aa),'IC',np.nanmean(aa))
df.to_csv('scripts/miner_3_20300311_trend_consistency_signal.csv',index=False)
ic.rename('ic').to_csv('scripts/miner_3_20300311_trend_consistency_ic.csv')
