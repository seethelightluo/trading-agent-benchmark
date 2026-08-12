import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=None
 for fn in(get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except: x=None
  if x is not None and len(x):break
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();lr=np.log(p).diff();r5=np.log(p/p.shift(5));v20=lr.rolling(20).std()*np.sqrt(252);disp=lr.rolling(5).std().mean(axis=1);thr=disp.rolling(252,min_periods=100).quantile(.75);gate=(disp>=thr).astype(float)
f=(-r5/v20).mul(gate,axis=0).shift(1);fr=np.log(p.shift(-10)/p);rows=[]
for dt in f.index:
 a=f.loc[dt];b=fr.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1:rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('shape',p.shape,'dates',len(z),'assets',len(D),'avgN',z.n.mean(),'coverage',z.n.mean()/len(D));print('H10 IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 x=q.loc[lo:hi];print(lo,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>2 else np.nan)
x=q.tail(120);print('recent120',len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>2 else np.nan);print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean());f.to_csv('scripts/miner_2_20311016_highdisp_reversal_signal.csv')
