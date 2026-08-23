import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-09-04')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}; p=pd.DataFrame({s:d[d.index<=cut] for s,d in px.items()}).sort_index(); r=p.pct_change(); f=p.pct_change(5).div(r.rolling(20,min_periods=15).std(),axis=0)
a=[]; ns=[]; cv=[]; tr=[]; ds=[]; ss=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  a.append(z);ns.append(ok.sum());cv.append(ok.mean());ds.append(p.index[i]);ss.append(x)
  if len(ss)>1:
   q=ss[-2];oo=x.notna()&q.notna();tr.append((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean())
a=np.array(a);D=pd.DatetimeIndex(ds); print({'factor':'momentum_5d_volnorm','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cv)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(tr))})
for n,m in [('180',D>=pd.Timestamp('2030-03-04')),('360',D>=pd.Timestamp('2029-09-04')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[m];print(n,len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)))
pd.DataFrame({'date':ds,'ic':a}).to_csv('scripts/miner_2_20300905_mom5_ic.csv',index=False);pd.DataFrame(ss,index=ds,columns=U).to_csv('scripts/miner_2_20300905_mom5_signal.csv');np.savez('scripts/miner_2_20300905_mom5_artifact.npz',dates=np.array(ds,dtype=str),signals=np.array(ss),assets=np.array(U))
