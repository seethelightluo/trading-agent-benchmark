import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={};H={};L={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=3000)
 if d is not None:
  z=d.set_index('date'); C[s]=z.close.astype(float); H[s]=z.high.astype(float); L[s]=z.low.astype(float)
p=pd.DataFrame(C).sort_index().ffill(); hi=pd.DataFrame(H).reindex(p.index).ffill(); lo=pd.DataFrame(L).reindex(p.index).ffill(); r=p.pct_change()
rng=(hi-lo).div(p.shift(1)).replace(0,np.nan); clv=((2*p-hi-lo).div((hi-lo).replace(0,np.nan))).clip(-1,1)
# Five-day exponentially smoothed candle pressure, volatility normalized and cross-sectionally demeaned.
pressure=(clv*rng).ewm(span=5,min_periods=5,adjust=False).mean()
f=(-pressure.sub(pressure.mean(axis=1),axis=0)).div(r.rolling(20).std())
def ic_series(h):
 y=p.pct_change(h).shift(-h); rows=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   c=z.f.corr(z.y)
   if np.isfinite(c): rows.append((p.index[i],c,len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def ir(x): return x.mean()/x.std(ddof=1) if len(x)>1 and x.std(ddof=1)>0 else np.nan
q=ic_series(1)
print('dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(ir(q.ic),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True).diff().abs().mean().mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031'),('2030','2032')]:
 z=q.loc[a:b].ic; print('regime',a,b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(ir(z),6) if len(z) else None)
for h in [1,3,5,10]:
 z=ic_series(h); print('decay',h,'dates',len(z),'IC',round(z.ic.mean(),6) if len(z) else None)
f.to_csv('scripts/miner_2_20320122_ewm_candle_pressure_signal.csv')
