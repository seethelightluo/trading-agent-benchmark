import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
TODAY='2029-12-17'; H=10
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 d=None
 try:d=get_index_daily_data(s,days=4000)
 except:pass
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,days=4000)
  except: d=None
 if d is not None and len(d):px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); P=P.loc[:TODAY]
rows=[]
for i in range(100,len(P)-H):
 date=P.index[i]
 for s in u:
  x=P[s].iloc[:i+1].dropna()
  if len(x)<100 or i+H>=len(P) or pd.isna(P[s].iloc[i+H]):continue
  ret5=x.iloc[-1]/x.iloc[-6]-1
  v5=x.pct_change().iloc[-6:-1].std(); v20=x.pct_change().iloc[-21:-1].std()
  if np.isfinite(ret5) and np.isfinite(v5) and np.isfinite(v20) and v5>0 and v20>0:
   # Compression breakout: recent direction is rewarded when short volatility expands from a compressed baseline.
   f=ret5*(v20/v5)
   rows.append((date,s,f,P[s].iloc[i+H]/x.iloc[-1]-1))
df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna()
ics=df.groupby('date').apply(lambda z:z.factor.corr(z.fwd),include_groups=False).dropna()
r=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('dates',len(ics),'avg_names',df.groupby('date').size().mean(),'coverage',df.symbol.nunique()/len(u),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(),'hit',(ics>0).mean(),'turnover',r.diff().abs().mean(axis=1).dropna().mean())
for h in [5,10,20,40]:
 aa=[]
 for i in range(100,len(P)-h):
  z=[]
  for s in u:
   x=P[s].iloc[:i+1].dropna()
   if len(x)<100 or i+h>=len(P) or pd.isna(P[s].iloc[i+h]):continue
   a=x.pct_change().iloc[-6:-1].std(); b=x.pct_change().iloc[-21:-1].std(); rr=x.iloc[-1]/x.iloc[-6]-1
   if a>0 and b>0:z.append((rr*b/a,P[s].iloc[i+h]/x.iloc[-1]-1))
  if len(z)>=8:aa.append(pd.Series([q[0] for q in z]).corr(pd.Series([q[1] for q in z])))
 print('horizon',h,'IC',np.nanmean(aa),'dates',len(aa))
df.to_csv('scripts/miner_3_20291217_compression_breakout_signal.csv',index=False)
