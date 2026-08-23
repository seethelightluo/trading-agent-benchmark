import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-05-30'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float); px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Persistence momentum: 30d trend risk-adjusted by realized volatility and fraction of up days.
r30=p.pct_change(30); v30=r.rolling(30,min_periods=20).std(); persist=(r>0).rolling(30,min_periods=20).mean()
fac=(r30/v30)*(0.5+persist)
ics=[]; dates=[]; ns=[]; cov=[]; turns=[]
for i in range(len(p)-10):
 dt=p.index[i]; end=p.index[i+10]
 if end>cut or dt<p.index[35]: continue
 x=fac.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z): ics.append(z); dates.append(dt); ns.append(ok.sum()); cov.append(ok.mean())
 if i>0:
  a=x.rank(pct=True); b=fac.iloc[i-1].rank(pct=True); oo=a.notna()&b.notna()
  if oo.sum(): turns.append((a[oo]-b[oo]).abs().mean())
a=np.array(ics); dates=np.array(dates,dtype='datetime64[ns]')
def sub(mask):
 q=a[mask]; return (len(q),float(q.mean()) if len(q) else None,float(q.mean()/q.std(ddof=1)) if len(q)>1 and q.std(ddof=1)>0 else None)
print({'factor':'persistence_adjusted_momentum_30d','dates':len(a),'start':str(pd.Timestamp(dates[0]).date()),'end':str(pd.Timestamp(dates[-1]).date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turns))})
for n,m in [('recent180',dates>=np.datetime64('2029-12-01')),('recent360',dates>=np.datetime64('2029-01-01')),('2028',(dates>=np.datetime64('2028-01-01'))&(dates<np.datetime64('2029-01-01'))),('2029',(dates>=np.datetime64('2029-01-01'))&(dates<np.datetime64('2030-01-01'))),('2030',dates>=np.datetime64('2030-01-01'))]: print(n,sub(m))
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_2_20300530_persistence_momentum_10d_signal.csv',index=False)
