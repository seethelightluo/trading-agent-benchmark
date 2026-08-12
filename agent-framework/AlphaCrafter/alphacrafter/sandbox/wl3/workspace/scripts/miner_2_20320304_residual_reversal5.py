import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'
p={a:pd.read_csv(f'{B}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in A}
dates=sorted(set.intersection(*[set(x.index) for x in p.values()])); ret=pd.DataFrame({a:p[a].reindex(dates).pct_change() for a in A})
ics=[]; signals=[]
for i,d in enumerate(dates):
 if i<31 or i+10>=len(dates): continue
 x=ret.iloc[i-20:i]; market=x.mean(axis=1); vals={}; fw={}
 for a in A:
  z=x[a]; ok=z.notna()&market.notna()
  if ok.sum()<12: continue
  beta=np.cov(z[ok],market[ok],ddof=1)[0,1]/np.var(market[ok],ddof=1) if np.var(market[ok],ddof=1)>1e-12 else 1
  # residual cumulative move over last 5 sessions, reversed and volatility normalized
  resid=(z-beta*market).iloc[-5:].sum(); vol=z.iloc[-20:].std()
  if np.isfinite(resid) and np.isfinite(vol) and vol>0:
   vals[a]=-resid/vol
   fw[a]=np.log(p[a].iloc[i+10]/p[a].iloc[i])
 if len(vals)>=8:
  ic=spearmanr(list(vals.values()),list(fw.values())).statistic; ics.append((d,ic,len(vals))); signals += [(d,a,vals[a],fw[a]) for a in vals]
d=pd.DataFrame(ics,columns=['date','ic','n']); print('dates',len(d),'avgN',d.n.mean(),'coverage',d.n.mean()/15,'IC',d.ic.mean(),'ICIR',d.ic.mean()/d.ic.std(ddof=1),'hit',(d.ic>0).mean())
for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030),(2031,2032)]:
 q=d[(d.date.dt.year>=lo)&(d.date.dt.year<=hi)].ic; print(lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('recent60',d.ic.tail(60).mean(),'recent120',d.ic.tail(120).mean())
out='scripts/miner_2_20320304_residual_reversal5'; pd.DataFrame(signals,columns=['date','asset','signal','fwd10']).to_csv(out+'_signal.csv',index=False); d.to_csv(out+'_ic.csv',index=False)
