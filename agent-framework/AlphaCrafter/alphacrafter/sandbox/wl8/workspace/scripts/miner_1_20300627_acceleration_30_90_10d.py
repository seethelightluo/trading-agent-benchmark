import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-06-27'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float); px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); r30=p.pct_change(30); r90=p.pct_change(90); v=r.rolling(30,min_periods=20).std().shift(1)
fac=((r30-0.5*r90)/(v+1e-8)).clip(-10,10); ics=[]; dates=[]; ns=[]; cov=[]; turns=[]
for i in range(100,len(p)-10):
 if p.index[i+10]>cut: continue
 x=fac.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 q=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(q): ics.append(q);dates.append(p.index[i]);ns.append(ok.sum());cov.append(ok.mean())
 if i:
  a=x.rank(pct=True);b=fac.iloc[i-1].rank(pct=True);o=a.notna()&b.notna()
  if o.sum():turns.append((a[o]-b[o]).abs().mean())
ics=np.array(ics);dates=pd.DatetimeIndex(dates); print({'factor':'acceleration_30_90_volnorm','dates':len(ics),'start':str(dates[0].date()),'end':str(dates[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(ics.mean()),'icir':float(ics.mean()/ics.std(ddof=1)),'hit':float((ics>0).mean()),'turnover':float(np.mean(turns))})
for n,m in [('recent180',dates>=pd.Timestamp('2029-12-27')),('recent360',dates>=pd.Timestamp('2029-01-01')),('2028',(dates>=pd.Timestamp('2028-01-01'))&(dates<pd.Timestamp('2029-01-01'))),('2029',(dates>=pd.Timestamp('2029-01-01'))&(dates<pd.Timestamp('2030-01-01'))),('2030',dates>=pd.Timestamp('2030-01-01'))]:
 z=ics[m];print(n,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':ics}).to_csv('scripts/miner_1_20300627_acceleration_30_90_10d_ic.csv',index=False)
pd.DataFrame({'date':dates}).to_csv('scripts/miner_1_20300627_acceleration_30_90_10d_signal.csv',index=False)
