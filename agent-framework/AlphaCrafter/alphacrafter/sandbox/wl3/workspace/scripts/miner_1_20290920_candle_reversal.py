import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,3200)
 if d is None or len(d)<200:d=get_index_daily_data(s,3200)
 if d is not None and len(d)>200:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date')
x=pd.DataFrame({s:d.close.astype(float) for s,d in P.items()}).sort_index().ffill();r=np.log(x).diff()
# Reversal weighted by recent candle pressure: fade 3d return, amplify when closes sit near range extremes.
raw=pd.DataFrame(index=x.index,columns=x.columns,dtype=float)
for s,d in P.items():
 q=d.reindex(x.index).ffill(); rng=(q.high-q.low)/q.close
 pressure=((q.close-q.open)/(q.high-q.low+1e-12)).clip(-1,1)
 raw[s]=(-r[s].rolling(3).sum())*(1+0.75*pressure.rolling(5).mean())/(r[s].rolling(60).std()*np.sqrt(3)+1e-12)
f=raw.sub(raw.mean(axis=1),axis=0).replace([np.inf,-np.inf],np.nan)
def run(h):
 y=np.log(x.shift(-h)/x);o=[]
 for dt in f.index:
  a=f.loc[dt];b=y.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8:o.append((dt,a[ok].corr(b[ok],method='spearman'),ok.sum()))
 return pd.DataFrame(o,columns=['date','ic','n']).dropna()
for h in [1,3,5,10]:
 a=run(h);print('H',h,'dates',len(a),'avgN',a.n.mean(),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
 if h==5:
  for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2029)]:
   q=a[(a.date.dt.year>=lo)&(a.date.dt.year<=hi)];print('REG',lo,hi,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
print('coverage',f.notna().sum().sum()/(f.shape[0]*15),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'assets',len(P),'dates',len(f))
f.to_csv('scripts/miner_1_20290920_candle_reversal_signal.csv',index_label='date')
