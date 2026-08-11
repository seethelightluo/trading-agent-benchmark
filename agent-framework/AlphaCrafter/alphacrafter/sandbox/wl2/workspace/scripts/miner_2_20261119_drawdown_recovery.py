import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in D.values()]))
P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change()
# drawdown recovery: location in recent range, conditioned on positive short-term slope
hi=P.rolling(80,min_periods=60).max(); lo=P.rolling(80,min_periods=60).min()
range_pos=(P-lo)/(hi-lo+1e-12)
F=(range_pos + 0.35*P.pct_change(20)).shift(1)
for h in [1,3,5,10,20]:
 Y=P.pct_change(h).shift(-h); q=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:
   v=spearmanr(z.f,z.y).statistic
   if np.isfinite(v):q.append(v);ns.append(len(z));ds.append(dt)
 x=np.array(q); print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
 if h==10:
  for lo_y,hi_y in [(2020,2021),(2022,2023),(2024,2025),(2026,2026)]:
   a=x[[d.year>=lo_y and d.year<=hi_y for d in ds]]; print('regime',lo_y,hi_y,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6))
  for n in [63,126,252,504]:
   a=x[-n:];print('recent',n,round(a.mean(),6),round(a.mean()/a.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
