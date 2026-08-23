import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-10-16'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']; px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); ret20=p.pct_change(20); broad=p.pct_change(60).median(axis=1)
# Smoothed regime: 20-day average of the cross-asset 60d median; reversal in positive regimes and trend-following in negative regimes.
reg=np.sign(broad.rolling(20,min_periods=10).mean()).replace(0,np.nan).ffill()
f=ret20.mul(-reg,axis=0)
ics=[]; ns=[]; cov=[]; turns=[]; dates=[]; sig=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=pd.Series(f.iloc[i],index=U); y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  ics.append(z); ns.append(ok.sum()); cov.append(ok.mean()); dates.append(p.index[i]); sig.append(x)
  if len(sig)>1:
   q=sig[-2]; oo=x.notna()&q.notna(); turns.append((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean())
a=np.array(ics); D=pd.DatetimeIndex(dates)
print({'factor':'smoothed_regime_reversal_20d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turns))})
for name,m in [('90',D>=pd.Timestamp('2030-07-01')),('180',D>=pd.Timestamp('2030-04-01')),('360',D>=pd.Timestamp('2029-10-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[m]; print(name,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_1_20301017_smoothed_regime_reversal_20d_ic.csv',index=False)
pd.DataFrame(sig,index=dates,columns=U).to_csv('scripts/miner_1_20301017_smoothed_regime_reversal_20d_signal.csv')
