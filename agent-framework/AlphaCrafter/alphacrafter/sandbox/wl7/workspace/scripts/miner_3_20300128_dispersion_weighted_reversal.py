import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
TODAY='2030-01-28'; u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 d=None
 try:d=get_index_daily_data(s,days=4000)
 except: pass
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,days=4000)
  except: d=None
 if d is not None and len(d): px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]
def calc(i,h):
 vals=[]
 for s in u:
  if s not in P: continue
  x=P[s].iloc[:i+1].dropna()
  if len(x)<80 or i+h>=len(P) or pd.isna(P[s].iloc[i+h]): continue
  rr=x.pct_change(); v=rr.iloc[-21:].std()
  if v<=0: continue
  rev=-(x.iloc[-1]/x.iloc[-4]-1)/v
  vals.append((s,rev,P[s].iloc[i+h]/x.iloc[-1]-1))
 if len(vals)<8:return None
 # high dispersion regime: cross-sectional dispersion of 20d returns, normalized by its trailing median
 r20=[]
 for s in u:
  if s in P:
   x=P[s].iloc[:i+1].dropna()
   if len(x)>=25:r20.append(x.iloc[-1]/x.iloc[-21]-1)
 disp=np.std(r20)
 hist=[]
 for k in range(max(80,i-250),i):
  q=[]
  for s in u:
   if s in P:
    x=P[s].iloc[:k+1].dropna()
    if len(x)>=25:q.append(x.iloc[-1]/x.iloc[-21]-1)
  if len(q)>=8:hist.append(np.std(q))
 gate=np.clip(disp/(np.median(hist) if hist else disp),0.5,2.0)
 return [(d, s, f*gate, fw) for s,f,fw in vals],gate
for h in [1,5,10,20]:
 rows=[]; gates=[]
 for i in range(100,len(P)-h):
  z=calc(i,h)
  if z:
   rows+= [(P.index[i],s,f,fw) for _,s,f,fw in z[0]]; gates.append(z[1])
 df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna()
 ic=df.groupby('date').apply(lambda z:z.factor.corr(z.fwd),include_groups=False).dropna()
 print('horizon',h,'dates',len(ic),'avg_names',df.groupby('date').size().mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'coverage',df.symbol.nunique()/len(u),'gate_mean',np.mean(gates))
 if h==10:
  r=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
  print('turnover',r.diff().abs().mean(axis=1).dropna().mean()); df.to_csv('scripts/miner_3_20300128_dispersion_weighted_reversal_signal.csv',index=False)
