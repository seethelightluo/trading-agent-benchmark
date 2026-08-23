import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-07-24')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
 px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
r20=p.pct_change(20); r60=p.pct_change(60)
# Adaptive trend: follow 20d asset momentum in a positive broad regime, fade it in a negative broad regime.
# Regime is median cross-asset 60d return, lagged contemporaneously at decision cutoff.
reg=(r60.median(axis=1)>0).astype(float)*2-1
fac=(r20/vol)*reg[:,None] if False else (r20.div(vol)).mul(reg,axis=0)
ics=[]; dates=[]; ns=[]; cov=[]; turns=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-05-01') or p.index[i+10]>cut: continue
 x=fac.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 q=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(q):
  ics.append(q); dates.append(p.index[i]); ns.append(ok.sum()); cov.append(ok.mean())
  if i>0:
   a=x.rank(pct=True); b=fac.iloc[i-1].rank(pct=True); oo=a.notna()&b.notna()
   if oo.sum(): turns.append((a[oo]-b[oo]).abs().mean())
a=np.array(ics); dates=np.array(dates,dtype='datetime64[ns]')
print({'factor':'regime_adaptive_momentum_20d','dates':len(a),'start':str(pd.Timestamp(dates[0]).date()),'end':str(pd.Timestamp(dates[-1]).date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turns))})
for n,m in [('recent180',dates>=np.datetime64('2030-01-01')),('recent360',dates>=np.datetime64('2029-07-24')),('2028',(dates>=np.datetime64('2028-01-01'))&(dates<np.datetime64('2029-01-01'))),('2029',(dates>=np.datetime64('2029-01-01'))&(dates<np.datetime64('2030-01-01'))),('2030',dates>=np.datetime64('2030-01-01'))]:
 q=a[m]; print(n,len(q),float(q.mean()) if len(q) else None,float(q.mean()/q.std(ddof=1)) if len(q)>1 else None)
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_2_20300725_regime_adaptive_momentum_20d_ic.csv',index=False)
# Signal artifact for deterministic recovery
np.savez('scripts/miner_2_20300725_regime_adaptive_momentum_20d_artifact.npz',dates=dates,ic=a)
