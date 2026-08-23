import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-05-16'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'];px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); bench=r.mean(axis=1)
# residual trend: 30d cumulative return after removing rolling 60d beta to cross-asset benchmark
beta=r.rolling(60,min_periods=40).cov(bench).div(bench.rolling(60,min_periods=40).var(),axis=0)
res=r.sub(beta.mul(bench,axis=0)); f=res.rolling(30,min_periods=20).sum()
ics=[]; ns=[]; cs=[]; ts=[]; ds=[]
for i in range(len(p)-10):
 dt=p.index[i];
 if dt<p.index[70] or p.index[i+10]>cut:continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 v=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(v):ics.append(v);ns.append(ok.sum());ds.append(dt);cs.append(ok.mean())
 if i:
  a=x.rank(pct=True);b=f.iloc[i-1].rank(pct=True);oo=a.notna()&b.notna();
  if oo.sum():ts.append((a[oo]-b[oo]).abs().mean())
a=np.array(ics);print({'factor':'beta_neutral_residual_trend_30d','dates':len(a),'start':str(ds[0].date()),'end':str(ds[-1].date()),'avg_instruments':np.mean(ns),'coverage':np.mean(cs),'ic':np.mean(a),'icir':np.mean(a)/np.std(a,ddof=1),'hit':np.mean(a>0),'turnover':np.mean(ts)})
for n,m in [('180',np.array(ds)>=pd.Timestamp('2029-11-18')),('360',np.array(ds)>=pd.Timestamp('2028-12-01')),('2029',(np.array(ds)>=pd.Timestamp('2029-01-01'))&(np.array(ds)<pd.Timestamp('2030-01-01'))),('2030',np.array(ds)>=pd.Timestamp('2030-01-01'))]:
 z=a[m];print(n,len(z),np.mean(z) if len(z) else None,np.mean(z)/np.std(z,ddof=1) if len(z)>1 else None)
pd.DataFrame({'date':ds,'ic':a}).to_csv('scripts/miner_3_20300516_residual_trend_10d_signal.csv',index=False)
