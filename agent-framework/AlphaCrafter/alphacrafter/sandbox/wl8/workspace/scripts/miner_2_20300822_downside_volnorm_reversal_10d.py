import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2030-08-21');P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date');P[s]=d[d.index<=cut].close
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); downside=r.where(r<0,0).rolling(20,min_periods=15).std(); f=-p.pct_change(10).div(downside,axis=0)
ics=[];ns=[];cv=[];tu=[];dates=[];sig=[]
for i in range(len(p)-10):
 if i<100 or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;o=x.notna()&y.notna()
 if o.sum()<8:continue
 z=spearmanr(x[o],y[o]).statistic
 if np.isfinite(z):
  ics.append(z);ns.append(o.sum());cv.append(o.mean());dates.append(p.index[i]);sig.append(x)
  if len(sig)>1:
   q=sig[-2];oo=x.notna()&q.notna();tu.append(float((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean()))
a=np.array(ics);dt=pd.DatetimeIndex(dates);print({'factor':'downside_volnorm_reversal_10d','dates':len(a),'start':str(dt[0].date()),'end':str(dt[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cv)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(tu))})
for n,m in [('180',dt>=pd.Timestamp('2030-01-01')),('360',dt>=pd.Timestamp('2029-08-01')),('2029',(dt>=pd.Timestamp('2029-01-01'))&(dt<pd.Timestamp('2030-01-01'))),('2030',dt>=pd.Timestamp('2030-01-01'))]:
 z=a[m];print(n,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
