import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}; V={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: x=fn(s,days=5000)
  except Exception: x=None
  if x is not None and len(x): break
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); x=x.sort_values('date').drop_duplicates('date').set_index('date'); P[s]=x.close.astype(float); V[s]=x.volume.astype(float) if 'volume' in x else pd.Series(index=x.index,dtype=float)
p=pd.DataFrame(P).sort_index().ffill(); vol=pd.DataFrame(V).reindex(p.index).ffill(); lr=np.log(p).diff()
r10=np.log(p/p.shift(10)); rv=lr.rolling(30).std()*np.sqrt(252); liq=(vol.rolling(10).mean()/vol.rolling(30).mean()).replace([np.inf,-np.inf],np.nan)
# Volume-confirmed, volatility-normalized medium momentum, lagged one session.
f=(r10/rv*liq).shift(1); fr=np.log(p.shift(-10)/p); rows=[]
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('shape',p.shape,'dates',len(z),'assets',len(P),'avgN',z.n.mean(),'coverage',z.n.mean()/len(P))
print('H10 IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 x=q.loc[lo:hi]; print(lo,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>2 else np.nan)
x=q.tail(120); print('recent120',len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>2 else np.nan)
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_1_20311002_volume_confirmed_momentum_signal.csv')
