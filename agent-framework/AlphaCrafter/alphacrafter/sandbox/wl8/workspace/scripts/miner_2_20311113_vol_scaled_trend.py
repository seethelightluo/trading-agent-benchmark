import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2031-10-30')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();r=p.pct_change(); v=r.rolling(20,min_periods=10).std().clip(lower=.002); f=(r.rolling(30,min_periods=20).sum()/v).shift(1)
D=[];A=[];N=[];C=[];T=[];pr=None
for i in range(len(p)-10):
 d=p.index[i]
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(d);A.append(z);N.append(ok.sum());C.append(ok.mean())
  if pr is not None:
   q=x.notna()&pr.notna();T.append(float((x[q].rank(pct=True)-pr[q].rank(pct=True)).abs().mean()))
  pr=x
D=pd.DatetimeIndex(D);a=np.array(A);print({'factor':'vol_scaled_trend_30d','dates':len(a),'avg_instruments':np.mean(N),'coverage':np.mean(C),'ic':a.mean(),'icir':a.mean()/a.std(ddof=1),'hit':(a>0).mean(),'turnover':np.mean(T)})
for n,m in [('365',D>=pd.Timestamp('2030-10-30')),('180',D>=pd.Timestamp('2031-04-30')),('60',D>=pd.Timestamp('2031-08-30')),('2030',(D>=pd.Timestamp('2030-01-01'))&(D<pd.Timestamp('2031-01-01'))),('2031',D>=pd.Timestamp('2031-01-01'))]:
 z=a[m];print(n,len(z),z.mean() if len(z) else np.nan,z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in D:
  x=f.loc[d];y=yy.loc[d];ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.nanmean(q))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_2_20311113_vol_scaled_trend_signal.csv');pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_2_20311113_vol_scaled_trend_ic.csv',index=False)
