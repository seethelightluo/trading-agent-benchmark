import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-08-07'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']
 px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Lower-turnover shock reversal: fade the trailing 10-session return, risk-normalized.
f=-p.pct_change(10).div(vol,axis=0)
ics=[]; ns=[]; cov=[]; turns=[]; dates=[]; sig=[]
for i in range(len(p)-10):
 if i<80 or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  ics.append(z); ns.append(int(ok.sum())); cov.append(float(ok.mean())); dates.append(p.index[i]); sig.append(x)
  if len(sig)>1:
   q=sig[-2]; oo=x.notna()&q.notna(); turns.append(float((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean()))
a=np.array(ics); D=pd.DatetimeIndex(dates)
print({'factor':'volnorm_shock_reversal_10d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turns))})
for name,m in [('180',D>=pd.Timestamp('2030-01-01')),('360',D>=pd.Timestamp('2029-08-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[m]; print(name,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_2_20300808_shock_reversal_10d_volnorm_ic.csv',index=False)
pd.DataFrame(sig,index=dates,columns=U).to_csv('scripts/miner_2_20300808_shock_reversal_10d_volnorm_signal.csv')
np.savez('scripts/miner_2_20300808_shock_reversal_10d_volnorm_artifact.npz',dates=np.array([str(x.date()) for x in dates]),signal=np.array(sig),assets=np.array(U))
