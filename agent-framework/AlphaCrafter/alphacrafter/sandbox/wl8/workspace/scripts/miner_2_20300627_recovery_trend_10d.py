import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-06-27'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float); px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); v20=r.rolling(20,min_periods=15).std(); r10=p.pct_change(10); lo60=p.rolling(60,min_periods=40).min(); hi60=p.rolling(60,min_periods=40).max(); pos=((p-lo60)/(hi60-lo60)).clip(0,1); fac=(r10/v20)*(0.5+pos)
ics=[];ds=[];ns=[];co=[];tr=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-03-01') or p.index[i+10]>cut: continue
 x=fac.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 q=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(q):
  ics.append(q);ds.append(p.index[i]);ns.append(ok.sum());co.append(ok.mean())
  if i:
   a=x.rank(pct=True);b=fac.iloc[i-1].rank(pct=True);oo=a.notna()&b.notna()
   if oo.sum():tr.append((a[oo]-b[oo]).abs().mean())
a=np.array(ics);ds=np.array(ds,dtype='datetime64[ns]'); print({'factor':'recovery_position_trend_10d','dates':len(a),'start':str(pd.Timestamp(ds[0]).date()),'end':str(pd.Timestamp(ds[-1]).date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(co)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(tr))})
for n,m in [('recent180',ds>=np.datetime64('2030-01-01')),('recent360',ds>=np.datetime64('2029-06-27')),('2028',(ds>=np.datetime64('2028-01-01'))&(ds<np.datetime64('2029-01-01'))),('2029',(ds>=np.datetime64('2029-01-01'))&(ds<np.datetime64('2030-01-01'))),('2030',ds>=np.datetime64('2030-01-01'))]:
 q=a[m];print(n,len(q),float(q.mean()) if len(q) else None,float(q.mean()/q.std(ddof=1)) if len(q)>1 else None)
pd.DataFrame({'date':ds,'ic':a}).to_csv('scripts/miner_2_20300627_recovery_trend_10d_signal.csv',index=False);np.savez('scripts/miner_2_20300627_recovery_trend_10d_artifact.npz',dates=ds,ic=a)
