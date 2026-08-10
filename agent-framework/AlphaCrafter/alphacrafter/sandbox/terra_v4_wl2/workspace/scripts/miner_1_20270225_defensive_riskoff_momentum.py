import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; EQ=U[:8]; DEF=['XAU','US10Y','CN10Y']
def load(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=5000)
   if x is not None:return x
  except: pass
px=pd.DataFrame({s:load(s).set_index('date').close for s in U}).sort_index(); r=px.pct_change();
# lag breadth: fraction of equity markets down over prior day, avoiding lookahead
breadth=r[EQ].lt(0).sum(axis=1).div(r[EQ].notna().sum(axis=1)).shift(1)
mom=r.rolling(5).sum(); med=mom.median(axis=1); high=(breadth>=.625)
# risk-off defensive momentum tilt; all names retain base score
sig=mom.copy()
for s in DEF: sig[s]=mom[s] + 2.0*high*(mom[s]-med)
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(sig.notna().sum().sum()/(len(U)*len(sig)),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028')]:
  ix=[i for i,d in enumerate(sig.index) if lo<=str(d)[:4]<=hi]
  aa=a[[j for j,d in enumerate([d for d in sig.index if True]) if False]] if False else None
  # recompute regime cleanly
  vv=[]
  for d in sig.index:
   if not (lo<=str(d)[:4]<=hi): continue
   z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1: vv.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  if len(vv)>2: print(' regime',lo,hi,'n',len(vv),'IC',round(np.mean(vv),6),'ICIR',round(np.mean(vv)/np.std(vv,ddof=1),6))
print('active fraction',high.mean())
