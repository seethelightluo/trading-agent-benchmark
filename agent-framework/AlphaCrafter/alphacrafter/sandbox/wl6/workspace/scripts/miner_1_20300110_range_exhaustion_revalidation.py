import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150:d=get_index_daily_data(s,2600)
 if d is not None:px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); out=[]
for i in range(90,len(P)-20):
 c=P.iloc[:i+1]; r=c.iloc[-1]/c.iloc[-11]-1; v=c.pct_change().iloc[-21:].std()*np.sqrt(252); hi=c.iloc[-61:].max();lo=c.iloc[-61:].min(); pos=(c.iloc[-1]-lo)/(hi-lo).replace(0,np.nan); f=(-r)*(0.5+pos)/v.replace(0,np.nan)
 for h in (1,5,10,20):
  fr=P.iloc[i+h]/P.iloc[i]-1; q=f.notna()&fr.notna()
  if q.sum()>=8:out.append((P.index[i],h,f[q].corr(fr[q]),q.mean(),q.sum()))
R=pd.DataFrame(out,columns=['date','h','ic','cov','n'])
for h in (1,5,10,20):
 x=R[R.h==h].dropna();z=x.ic;print(h,len(z),x.n.mean(),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),x["cov"].mean())
x=R[R.h==10].dropna();print('regimes')
for a,b in [('2020','2024'),('2025','2026'),('2027','2028'),('2029','2030')]:
 z=x[(x.date>=a)&(x.date<=b+'-12-31')];print(a,len(z),z.ic.mean() if len(z) else np.nan)
