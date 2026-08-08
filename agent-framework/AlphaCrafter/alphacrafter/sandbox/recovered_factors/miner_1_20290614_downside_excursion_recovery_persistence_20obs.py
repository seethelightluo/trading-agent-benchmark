import numpy as np,pandas as pd
from scipy.stats import spearmanr
# Candidate: downside-excursion recovery persistence. On each asset's negative intraday days,
# measure close location in its daily range; take a native 20-observation mean (min 10 events).
ROOT='../persistent/stock_data';A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END='2029-06-13'
D={a:pd.read_csv(f'{ROOT}/{a}.csv').set_index('date').sort_index().loc[:END] for a in A}
ix=sorted(set().union(*[set(d.index) for d in D.values()]));c=pd.DataFrame({a:D[a].reindex(ix).close for a in A})
F={}
for a,d in D.items():
 rng=(d.high-d.low).replace(0,np.nan); loc=(d.close-d.low)/rng
 # only a genuine close-to-open loss contributes; NaN observations do not become neutral
 recovery=loc.where(d.close<d.open)
 F[a]=recovery.rolling(20,min_periods=10).mean()
f=pd.DataFrame(F).reindex(ix);vis=c.index[c.index<=END]
print('FACTOR downside_excursion_recovery_persistence_20obs endpoint',vis[-1],'assets',len(A),'cells',int(f.notna().sum().sum()),'of',len(vis)*15)
def stat(sub,h):
 fw=c.shift(-h).div(c)-1; vals=[];ns=[];turn=[];last=None
 for t in sub:
  z=pd.concat([f.loc[t],fw.loc[t]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  q=f.loc[t].rank(); zz=pd.concat([q,last],axis=1).dropna() if last is not None else pd.DataFrame()
  if len(zz)>=8:turn.append(1-spearmanr(zz.iloc[:,0],zz.iloc[:,1]).statistic)
  last=q
 x=np.array(vals);return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns),np.mean(turn)
for h in [1,5,10,20]:
 x=stat(vis,h);print('H',h,'dates',x[0],'IC',round(x[1],6),'ICIR',round(x[2],6),'hit',round(x[3],4),'mean_n',round(x[4],2),'coverage',round(f.notna().mean().mean(),4),'turn',round(x[5],4))
for label,sub in [('2020_21',vis[vis<'2022-01-01']),('2022_23',vis[(vis>='2022-01-01')&(vis<'2024-01-01')]),('2024_25',vis[(vis>='2024-01-01')&(vis<'2026-01-01')]),('2026_current',vis[vis>='2026-01-01'])]:
 x=stat(sub,5);print('REGIME',label,'dates',x[0],'IC',round(x[1],6),'ICIR',round(x[2],6),'hit',round(x[3],4))
