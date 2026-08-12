import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except Exception:x=None
  if x is not None and len(x): break
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); lp=np.log(p); r=lp.diff(); rows=[]
# Volatility-normalized short reversal, lagged. Test 3-session reversal against 1/10d forward returns.
for look in [3,5,7]:
 vol=r.rolling(20,min_periods=12).std()*np.sqrt(20); f=(-(lp-lp.shift(look))/vol).shift(1)
 for h in [1,10]:
  fr=lp.shift(-h)-lp; out=[]
  for dt in f.index:
   a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
   if ok.sum()>=8 and a[ok].nunique()>1: out.append((dt,a[ok].corr(b[ok]),ok.sum()))
  z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); q=z.ic
  print('look',look,'H',h,'dates',len(q),'avgN',z.n.mean(),'coverage',z.n.mean()/len(D),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
  for lo,hi in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2031')]:
   x=q.loc[lo:hi]; print(lo,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),5))
  if look==5 and h==10:
   f.to_csv('scripts/miner_2_20311225_volnorm_reversal5_signal.csv'); z.to_csv('scripts/miner_2_20311225_volnorm_reversal5_ic.csv')
