import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().close for a in A}
r=pd.concat({a:p[a].pct_change() for a in A},axis=1); disp=r.rolling(20).std().mean(axis=1)
# high-dispersion activation, lagged percentile; directional efficiency with recent reversal emphasis
raw={}
for a in A:
 rr=p[a].pct_change(); eff=rr.rolling(20).sum()/(rr.abs().rolling(20).sum()+1e-12)
 raw[a]=eff
rows=[]; obs=[]
for d in sorted(set().union(*[set(x.index) for x in p.values()])):
 if d not in disp.index: continue
 # percentile computed from history through d (signal uses d close, decision convention)
 hist=disp.loc[:d].tail(252); q=hist.quantile(.70) if len(hist)>50 else np.nan
 act=1.0 if np.isfinite(q) and disp.loc[d]>q else 0.0
 vals={a:raw[a].get(d,np.nan)*act for a in A}; good=[v for v in vals.values() if np.isfinite(v)]
 for a in A: obs.append((d,a,vals[a]))
 for h in [5,10]:
  f=[];y=[]
  for a in A:
   if d not in p[a].index: continue
   i=p[a].index.get_loc(d);v=vals[a]
   if np.isfinite(v) and i+h<len(p[a]): f.append(v);y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8 and np.nanstd(f)>0: rows.append((d,h,spearmanr(f,y).statistic,len(f)))
x=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [5,10]:
 z=x[x.h==h];print('H',h,'dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.mean()/15,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic;print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=pd.DataFrame(obs,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_dispersion_trend.csv',index=False)
print('turnover',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean().mean())
