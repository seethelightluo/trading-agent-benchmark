import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-11-27')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();r=p.pct_change();d=r.std(axis=1).rolling(20,min_periods=15).mean();q=d.rolling(252,min_periods=126).quantile(.5);f=r.rolling(10,min_periods=10).sum().shift(1).mul((d<q).shift(1),axis=0)
D=[];A=[];N=[];T=[];prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(p.index[i]);A.append(z);N.append(ok.sum())
  if prev is not None:T.append(float((x[ok].rank(pct=True)-prev[ok].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D);A=np.array(A);print({'factor':'low_dispersion_trend_10d','dates':len(A),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(N)),'coverage':1.0,'ic':float(A.mean()),'icir':float(A.mean()/A.std(ddof=1)),'hit':float((A>0).mean()),'turnover':float(np.mean(T))})
for n,m in [('recent180',D>=pd.Timestamp('2031-01-01')),('recent360',D>=pd.Timestamp('2030-11-01')),('2028',(D>=pd.Timestamp('2028-01-01'))&(D<pd.Timestamp('2029-01-01'))),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',(D>=pd.Timestamp('2030-01-01'))&(D<pd.Timestamp('2031-01-01'))),('2031',D>=pd.Timestamp('2031-01-01'))]:
 z=A[m];print(n,len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)),float((z>0).mean()))
pd.DataFrame([f.loc[x].values for x in D],index=D,columns=U).to_csv('scripts/miner_3_20311127_low_dispersion_10d_signal.csv');pd.DataFrame({'date':D,'ic':A}).to_csv('scripts/miner_3_20311127_low_dispersion_10d_ic.csv',index=False)
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;z=[]
 for dd in D:
  x=f.loc[dd];y=yy.loc[dd];ok=x.notna()&y.notna();z.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,float(np.nanmean(z)))
