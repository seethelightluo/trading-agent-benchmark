import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2031-10-16')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();r=p.pct_change()
# Volatility-normalized short-term reversal, lagged one completed session.
f=(-p.pct_change(5)/(r.rolling(20,min_periods=10).std()*np.sqrt(5)).clip(lower=.003)).shift(1)
D=[];a=[];ns=[];cv=[];turn=[];prev=None
for i in range(len(p)-10):
 d=p.index[i]
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(d);a.append(z);ns.append(ok.sum());cv.append(ok.mean())
  if prev is not None:
   oo=x.notna()&prev.notna();turn.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D);a=np.array(a);print({'factor':'volatility_normalized_reversal_5d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cv)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turn))})
for name,mask in [('365',D>=pd.Timestamp('2030-10-16')),('180',D>=pd.Timestamp('2031-04-16')),('60',D>=pd.Timestamp('2031-07-16')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',(D>=pd.Timestamp('2030-01-01'))&(D<pd.Timestamp('2031-01-01'))),('2031YTD',D>=pd.Timestamp('2031-01-01'))]:
 z=a[mask];print(name,len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)))
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in D:
  x=f.loc[d];y=yy.loc[d];ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,float(np.nanmean(q)),len(q))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_2_20311030_volnorm_reversal_signal.csv');pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_2_20311030_volnorm_reversal_ic.csv',index=False)
