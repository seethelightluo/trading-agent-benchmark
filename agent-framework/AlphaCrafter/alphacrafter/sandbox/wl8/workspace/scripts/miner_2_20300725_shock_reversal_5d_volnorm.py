import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-07-24')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
p=pd.DataFrame({s:d[d.index<=cut] for s,d in px.items()}).sort_index(); r=p.pct_change(); v=r.rolling(20,min_periods=15).std()
# Short-horizon shock reversal, tempered by recent volatility: large recent losses are favored, but avoid extreme noisy names.
fac=-p.pct_change(5).div(v)
ics=[]; dates=[]; ns=[]; cov=[]; turns=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-03-01') or p.index[i+10]>cut: continue
 x=fac.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 q=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(q):
  ics.append(q); dates.append(p.index[i]); ns.append(ok.sum()); cov.append(ok.mean())
  if i>0:
   a=x.rank(pct=True); b=fac.iloc[i-1].rank(pct=True); oo=a.notna()&b.notna()
   if oo.sum(): turns.append((a[oo]-b[oo]).abs().mean())
a=np.array(ics); dates=np.array(dates,dtype='datetime64[ns]')
print({'factor':'shock_reversal_5d_volnorm','dates':len(a),'start':str(pd.Timestamp(dates[0]).date()),'end':str(pd.Timestamp(dates[-1]).date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turns))})
for n,m in [('recent180',dates>=np.datetime64('2030-01-01')),('recent360',dates>=np.datetime64('2029-07-24')),('2028',(dates>=np.datetime64('2028-01-01'))&(dates<np.datetime64('2029-01-01'))),('2029',(dates>=np.datetime64('2029-01-01'))&(dates<np.datetime64('2030-01-01'))),('2030',dates>=np.datetime64('2030-01-01'))]:
 q=a[m]; print(n,len(q),float(q.mean()) if len(q) else None,float(q.mean()/q.std(ddof=1)) if len(q)>1 else None)
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_2_20300725_shock_reversal_5d_volnorm_ic.csv',index=False)
np.savez('scripts/miner_2_20300725_shock_reversal_5d_volnorm_artifact.npz',dates=dates,ic=a)
