import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2031-10-16')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();r=p.pct_change();
# Risk-adjusted intermediate momentum, with cross-sectional volatility rank penalty.
f=p.pct_change(20)/(r.rolling(30).std()*np.sqrt(20)).clip(lower=.005)
D=[];a=[];ns=[];cov=[];prev=None;turn=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(p.index[i]);a.append(z);ns.append(int(ok.sum()));cov.append(float(ok.mean()))
  if prev is not None:
   oo=x.notna()&prev.notna();turn.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D);a=np.array(a);print({'factor':'risk_adjusted_intermediate_momentum_20d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turn))})
for n,m in [('180',D>=pd.Timestamp('2031-01-01')),('360',D>=pd.Timestamp('2030-04-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',(D>=pd.Timestamp('2030-01-01'))&(D<pd.Timestamp('2031-01-01'))),('recent60',D>=pd.Timestamp('2031-07-01'))]:
 z=a[m];print(n,len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)))
for h in [1,5,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in D:
  x=f.loc[d];y=yy.loc[d];ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,float(np.nanmean(q)),len(q))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_3_20311016_risk_adj_intermediate_signal.csv');pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_3_20311016_risk_adj_intermediate_ic.csv',index=False)
