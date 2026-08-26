import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; TODAY='2029-12-17'; H=20
try: U=get_account_dict().get('watch_list',[]) or U
except: pass
px={}
for s in U:
 try:d=get_index_daily_data(s,days=4000)
 except:d=None
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,days=4000)
  except:d=None
 if d is not None and len(d):px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]; rows=[]
for i in range(100,len(P)-H):
 z=[]
 for s in U:
  if s not in P:continue
  x=P[s].iloc[:i+1].dropna()
  if len(x)<80 or i+H>=len(P) or pd.isna(P[s].iloc[i+H]):continue
  r=x.pct_change().dropna(); r60=x.iloc[-1]/x.iloc[-61]-1; v=r.iloc[-61:-1].std()*np.sqrt(60)
  # trend strength penalized by volatility, with persistence: fraction positive days
  pers=(r.iloc[-61:-1]>0).mean(); sig=r60/(v+1e-12)*(0.5+pers)
  if np.isfinite(sig):z.append((s,sig,P[s].iloc[i+H]/x.iloc[-1]-1))
 if len(z)>=8:rows += [(P.index[i],)+q for q in z]
df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']);ic=df.groupby('date').apply(lambda q:q.factor.corr(q.fwd),include_groups=False).dropna(); ranks=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('dates',len(ic),'avg_names',df.groupby('date').size().mean(),'coverage',df.symbol.nunique()/len(U),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'turn',ranks.diff().abs().mean(axis=1).dropna().mean());
for j,(a,b) in enumerate([(0,len(ic)//3),(len(ic)//3,2*len(ic)//3),(2*len(ic)//3,len(ic))]):
 q=ic.iloc[a:b];print('regime',j,len(q),q.mean(),q.mean()/q.std())
