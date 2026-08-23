import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-01-08')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=p.pct_change(); rv=r.rolling(20).std(); base=-p.pct_change(10)/(rv*np.sqrt(10)).clip(lower=.005); tr20=p.pct_change(20); disp=tr20.std(axis=1); q=disp.rolling(120,min_periods=60).rank(pct=True); f=base.mul(.5+q,axis=0)
D=[];A=[];N=[];C=[];T=[];prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(p.index[i]);A.append(z);N.append(ok.sum());C.append(ok.mean())
  if prev is not None:
   oo=x.notna()&prev.notna();T.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D);a=np.array(A); print({'dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_n':round(np.mean(N),2),'coverage':round(np.mean(C),4),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1),6),'hit':round((a>0).mean(),4),'turnover':round(np.mean(T),6)})
for lab,dt in [('360','2030-01-01'),('180','2030-07-01'),('90','2030-10-01'),('60','2030-11-01')]:
 z=a[D>=pd.Timestamp(dt)];print(lab,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_3_20310109_dispersion_gated_ic.csv',index=False);pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_3_20310109_dispersion_gated_signal.csv')
