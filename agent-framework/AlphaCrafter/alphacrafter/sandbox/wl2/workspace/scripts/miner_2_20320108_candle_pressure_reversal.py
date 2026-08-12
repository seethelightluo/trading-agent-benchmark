import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={};H={};L={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=3000)
 if d is not None:
  z=d.set_index('date'); C[s]=z.close.astype(float); H[s]=z.high.astype(float); L[s]=z.low.astype(float)
p=pd.DataFrame(C).sort_index().ffill(); hi=pd.DataFrame(H).reindex(p.index).ffill(); lo=pd.DataFrame(L).reindex(p.index).ffill()
r=p.pct_change(); market=r.mean(axis=1); resid=r.sub(market,axis=0)
# Candle-pressure reversal: fade abnormal close-location pressure, neutralized to the cross-asset move,
# with range and 20d volatility normalization. All components use data through decision day.
rng=(hi-lo).div(p.shift(1)).replace(0,np.nan)
clv=((2*p-hi-lo).div((hi-lo).replace(0,np.nan))).clip(-1,1)
pressure=(clv*rng).rolling(3).sum()
f=(-pressure.sub(pressure.mean(axis=1),axis=0)).div(r.rolling(20).std())
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:
  c=z.f.corr(z.y)
  if np.isfinite(c): rows.append((p.index[i],c,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def ir(x): return x.mean()/x.std(ddof=1) if len(x)>1 and x.std(ddof=1)>0 else np.nan
print('dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(ir(q.ic),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True).diff().abs().mean().mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031'),('2030','2032')]:
 z=q.loc[a:b].ic; print('regime',a,b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(ir(z),6) if len(z) else None)
for h in [1,3,5,10]:
 y=p.pct_change(h).shift(-h); rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.corr(z.y))
 print('decay',h,'dates',len(rr),'IC',round(np.nanmean(rr),6) if rr else None)
f.to_csv('scripts/miner_2_20320108_candle_pressure_reversal_signal.csv')
