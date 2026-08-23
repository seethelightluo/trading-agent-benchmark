import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-01-22')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();r=p.pct_change()
# Cross-asset dispersion gate: momentum in low-dispersion trend regimes, reversal in high dispersion.
disp=r.std(axis=1).rolling(20,min_periods=15).mean(); med=disp.rolling(252,min_periods=126).median(); g=((disp/med-1).clip(0,1)).shift(1)
trend=r.rolling(20,min_periods=20).sum().shift(1); rev=-r.rolling(5,min_periods=5).sum().shift(1); f=trend.mul(1-g,axis=0)+rev.mul(g,axis=0)
D=[];A=[];N=[];C=[];T=[];prev=None
for i in range(len(p)-10):
 d=p.index[i]
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8 or x[ok].nunique()<3 or y[ok].nunique()<3:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(d);A.append(z);N.append(ok.sum());C.append(ok.mean())
  if prev is not None:
   oo=x.notna()&prev.notna();T.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D);A=np.array(A)
def st(m):
 z=A[m];return (len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round((z>0).mean(),4))
print({'factor':'dispersion_gated_trend_reversal','dates':len(A),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':round(np.mean(N),2),'coverage':round(np.mean(C),4),'ic':round(A.mean(),6),'icir':round(A.mean()/A.std(ddof=1),6),'hit':round((A>0).mean(),4),'turnover':round(np.mean(T),6)})
for n,m in [('365',D>=pd.Timestamp('2031-01-22')),('180',D>=pd.Timestamp('2031-07-01')),('60',D>=pd.Timestamp('2031-11-01')),('2028',(D>=pd.Timestamp('2028-01-01'))&(D<pd.Timestamp('2029-01-01'))),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',(D>=pd.Timestamp('2030-01-01'))&(D<pd.Timestamp('2031-01-01'))),('2031',(D>=pd.Timestamp('2031-01-01'))&(D<pd.Timestamp('2032-01-01')))]:print(n,st(m))
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in D:
  x=f.loc[d];y=yy.loc[d];ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,round(np.nanmean(q),6),len(q))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_2_20320122_dispersion_gated_signal.csv');pd.DataFrame({'date':D,'ic':A}).to_csv('scripts/miner_2_20320122_dispersion_gated_ic.csv',index=False)
