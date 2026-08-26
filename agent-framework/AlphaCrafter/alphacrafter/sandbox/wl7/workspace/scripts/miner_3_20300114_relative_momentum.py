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
  except:d=None
 if d is not None and len(d):px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]; rows=[]
for i in range(100,len(P)-H):
 vals=[]
 for s in u:
  x=P[s].iloc[:i+1].dropna()
  if len(x)<100 or i+H>=len(P) or pd.isna(P[s].iloc[i+H]):continue
  r=x.pct_change(); v=r.iloc[-61:-1].std()
  if v>0: vals.append((s,(x.iloc[-1]/x.iloc[-21]-1)/v))
 if len(vals)<8:continue
 med=np.median([v for _,v in vals])
 for s,v in vals:
  rows.append((P.index[i],s,v-med,P[s].iloc[i+H]/P[s].iloc[:i+1].dropna().iloc[-1]-1))
df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna(); ic=df.groupby('date').apply(lambda z:z.factor.corr(z.fwd),include_groups=False).dropna(); r=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('dates',len(ic),'avg_names',df.groupby('date').size().mean(),'coverage',df.symbol.nunique()/len(u),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'turnover',r.diff().abs().mean(axis=1).dropna().mean())
for h in [5,10,20,40]:
 a=[]
 for i in range(100,len(P)-h):
  z=[]
  for s in u:
   x=P[s].iloc[:i+1].dropna()
   if len(x)<100 or i+h>=len(P) or pd.isna(P[s].iloc[i+h]):continue
   rr=x.pct_change(); v=rr.iloc[-61:-1].std()
   if v>0:z.append(((x.iloc[-1]/x.iloc[-21]-1)/v,P[s].iloc[i+h]/x.iloc[-1]-1))
  if len(z)>=8:
   med=np.median([q[0] for q in z]); a.append(pd.Series([q[0]-med for q in z]).corr(pd.Series([q[1] for q in z])))
 print('horizon',h,'IC',np.nanmean(a),'dates',len(a))
df.to_csv('scripts/miner_3_20300114_relative_momentum_signal.csv',index=False)
