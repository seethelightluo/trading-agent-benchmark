import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-06-27'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float); px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); r30=p.pct_change(30)
down=r.where(r<0,0).rolling(30,min_periods=20).std(); neg=(r<0).rolling(30,min_periods=20).mean(); fac=(r30/(down+1e-8))*(1-0.35*neg)
ics=[]; dates=[]; ns=[]; cov=[]; turns=[]
for i in range(35,len(p)-10):
 dt=p.index[i]; end=p.index[i+10]
 if end>cut: continue
 x=fac.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z): ics.append(z); dates.append(dt); ns.append(ok.sum()); cov.append(ok.mean())
 if i>0:
  a=x.rank(pct=True); b=fac.iloc[i-1].rank(pct=True); oo=a.notna()&b.notna()
  if oo.sum(): turns.append((a[oo]-b[oo]).abs().mean())
ics=np.array(ics); dates=pd.DatetimeIndex(dates)
print({'factor':'downside_adjusted_momentum_30d','dates':len(ics),'start':str(dates[0].date()),'end':str(dates[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(ics.mean()),'icir':float(ics.mean()/ics.std(ddof=1)),'hit':float((ics>0).mean()),'turnover':float(np.mean(turns))})
for n,mask in [('recent180',dates>=pd.Timestamp('2029-12-27')),('recent360',dates>=pd.Timestamp('2029-01-01')),('2028',(dates>=pd.Timestamp('2028-01-01'))&(dates<pd.Timestamp('2029-01-01'))),('2029',(dates>=pd.Timestamp('2029-01-01'))&(dates<pd.Timestamp('2030-01-01'))),('2030',dates>=pd.Timestamp('2030-01-01'))]:
 z=ics[mask]; print(n,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':ics}).to_csv('scripts/miner_1_20300627_downside_adjusted_momentum_10d_ic.csv',index=False)
pd.DataFrame({'date':dates}).to_csv('scripts/miner_1_20300627_downside_adjusted_momentum_10d_signal.csv',index=False)
