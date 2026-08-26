import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>120:
  d=d.copy();d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index();R=P.pct_change(); V=R.rolling(40,min_periods=30).std(); sig=-(P.pct_change(40)/(np.sqrt(40)*V)).shift(1); sig=sig.sub(sig.mean(axis=1),axis=0)
for h in [1,5,10,20]:
 F=P.pct_change(h).shift(-h); rows=[]
 for dt in P.index:
  if dt<pd.Timestamp('2029-11-03'):continue
  a=sig.loc[dt];b=F.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8:rows.append(a[ok].corr(b[ok]))
 z=pd.Series(rows).dropna(); print(h,len(z),round(z.mean(),6),round(z.mean()/z.std(),6),round((z>0).mean(),4))
print('coverage',sig.loc['2029-11-03':].notna().mean().mean())
