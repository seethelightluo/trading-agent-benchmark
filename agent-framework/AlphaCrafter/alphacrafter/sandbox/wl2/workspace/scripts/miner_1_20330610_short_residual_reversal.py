import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-06-10')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); lr=np.log(p).diff()
# Candidate: short-term residual reversal: negative 3d return after removing cross-sectional mean, scaled by idiosyncratic 20d vol.
cs=lr.sub(lr.mean(axis=1),axis=0)
vol=lr.rolling(20,min_periods=12).std().shift(1)
fac=(-cs.rolling(3,min_periods=3).sum().shift(1)/vol).replace([np.inf,-np.inf],np.nan)
rows=[]
for h in [1,3,5,10]:
  ics=[]; ns=[]
  fr=np.log(p).shift(-h)-np.log(p)
  for dt in fac.index:
    a=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(a)>=8:
      ics.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); ns.append(len(a))
  x=np.asarray(ics); print(f'h={h} dates={len(x)} avgN={np.mean(ns):.3f} coverage={np.mean(ns)/15:.3f} IC={np.nanmean(x):.8f} ICIR={np.nanmean(x)/np.nanstd(x,ddof=1):.8f} hit={np.mean(x>0):.4f}')
  if h==1:
   out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20330610_short_residual_reversal_signal.csv',index=False)
# regime split
fr=np.log(p).shift(-1)-np.log(p); ics=[]
for dt in fac.index:
 a=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8: ics.append((dt,spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic))
x=pd.DataFrame(ics,columns=['date','ic']);
for name,z in [('2020-25',x[x.date<'2026-01-01']),('2026-29',x[(x.date>='2026-01-01')&(x.date<'2030-01-01')]),('2030-33',x[x.date>='2030-01-01'])]: print(name,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
print('turnover proxy',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
