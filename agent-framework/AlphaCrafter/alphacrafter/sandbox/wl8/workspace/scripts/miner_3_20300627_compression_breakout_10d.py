import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-06-27'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']; px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Compression-confirmed breakout: directional 20d return, scaled by volatility, with a bonus when
# the preceding 10d volatility was compressed relative to its 60d baseline.
vol20=r.rolling(20,min_periods=15).std(); vol60=r.rolling(60,min_periods=40).std()
compression=(vol60/vol20).clip(0.5,2.0)
mom20=p.pct_change(20)
f=(mom20/(vol20*np.sqrt(20))).mul(compression.clip(0.75,1.5))
ics=[];ns=[];cs=[];turn=[];ds=[]
for i in range(len(p)-10):
 dt=p.index[i]
 if dt<p.index[70] or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 v=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(v):
  ics.append(v);ns.append(ok.sum());cs.append(ok.mean());ds.append(dt)
  if i>0:
   prev=f.iloc[i-1]; oo=x.notna()&prev.notna();
   if oo.sum(): turn.append((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean())
a=np.array(ics); print({'factor':'compression_confirmed_breakout_20d','dates':len(a),'start':str(ds[0].date()),'end':str(ds[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cs)),'ic':float(np.mean(a)),'icir':float(np.mean(a)/np.std(a,ddof=1)),'hit':float(np.mean(a>0)),'turnover':float(np.mean(turn))})
D=np.array(ds)
for name,m in [('180',D>=pd.Timestamp('2029-12-01')),('360',D>=pd.Timestamp('2029-01-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[m]; print(name,len(z),float(np.mean(z)) if len(z) else None,float(np.mean(z)/np.std(z,ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':ds,'ic':a}).to_csv('scripts/miner_3_20300627_compression_breakout_10d_ic.csv',index=False)
# signal artifact for provenance
f.loc[pd.Index(ds)].to_csv('scripts/miner_3_20300627_compression_breakout_10d_signal.csv')
