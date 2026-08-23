import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Range exhaustion reversal, conditioned on broad market trend: fade weakness only when benchmark trend is positive,
# otherwise favor defensive relative strength through sign-preserving trend gate.
px={}
for s in U:
 d=get_stock_daily_data(s, days=2600)
 if d is None or len(d)<150: d=get_index_daily_data(s, days=2600)
 if d is not None: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill()
# use dates with complete enough universe, no future leakage
rows=[]
for i in range(90,len(P)-10):
 date=P.index[i]; cur=P.iloc[:i+1]
 r10=cur.iloc[-1]/cur.iloc[-11]-1
 r60=cur.iloc[-1]/cur.iloc[-61]-1
 vol=cur.pct_change().iloc[-21:].std()*np.sqrt(252)
 hi=cur.iloc[-61:].max(); lo=cur.iloc[-61:].min(); pos=(cur.iloc[-1]-lo)/(hi-lo).replace(0,np.nan)
 # market breadth trend gate computed only through date
 breadth=(r60>0).mean()
 gate=0.5+0.5*breadth
 f=(-r10)*(0.5+pos)*gate/vol.replace(0,np.nan)
 for h in [1,5,10,20]:
  if i+h<len(P):
   fr=P.iloc[i+h]/P.iloc[i]-1
   valid=f.notna()&fr.notna()
   if valid.sum()>=8: rows.append((date,h,f[valid].corr(fr[valid]),valid.mean(),valid.sum()))
R=pd.DataFrame(rows,columns=['date','h','ic','coverage','n'])
for h in [1,5,10,20]:
 x=R[R.h==h].dropna(); ic=x.ic
 print('horizon',h,'dates',len(x),'avg_n',x.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit', (ic>0).mean(),'coverage',x.coverage.mean())
x=R[R.h==10].dropna(); print('regimes')
for a,b in [('2020','2024'),('2025','2026'),('2027','2028'),('2029','2030')]:
 z=x[(x.date>=a)&(x.date<=b+'-12-31')]; print(a,b,len(z),z.ic.mean() if len(z) else np.nan)
