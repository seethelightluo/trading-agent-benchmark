import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-07-15')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).query('date<=@cutoff').set_index('date')['close'].reindex(r.index).ffill().pct_change()
# macro-neutral momentum: trailing 20d return less rolling beta to DXY times DXY trailing return
factors=[]; rows={1:[],5:[],10:[]}
for i in range(65,len(r)-10):
 rr=r.iloc[i-40:i]; m=macro.iloc[i-40:i]
 if m.notna().sum()<30: continue
 var=m.var(); beta=rr.apply(lambda x: x.cov(m)/var if var>1e-12 else np.nan)
 mom=r.iloc[i-20:i].sum(); dm=m.iloc[-20:].sum()
 f=mom-beta*dm
 y={h:r.iloc[i+1:i+1+h].sum() for h in rows}
 for h in rows:
  z=pd.concat([f,y[h]],axis=1).dropna()
  if len(z)>=8: rows[h].append((r.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 factors.append((r.index[i],f))
for h,x in rows.items():
 ic=pd.Series(dict(x)); print('H',h,'dates',len(ic),'avgN',r.loc[ic.index].notna().sum(axis=1).mean(),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=ic[(ic.index.year>=lo)&(ic.index.year<=hi)]; print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else np.nan)
q=pd.DataFrame({d:f for d,f in factors}).T.rank(axis=1,pct=True)
print('turnover',q.diff().abs().mean().mean(),'coverage',r.loc[q.index].notna().mean().mean(),'last',q.index[-1])
