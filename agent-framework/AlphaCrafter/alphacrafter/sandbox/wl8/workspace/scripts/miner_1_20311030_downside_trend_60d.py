import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2031-10-30')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();r=p.pct_change(); down=r.where(r<0,0).rolling(60).std().shift(1).clip(lower=.003); f=p.pct_change(60).shift(1)/down
D=[];a=[];ns=[];cov=[];turn=[];prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2021-01-01') or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(p.index[i]);a.append(z);ns.append(int(ok.sum()));cov.append(float(ok.mean()))
  if prev is not None:
   oo=x.notna()&prev.notna();turn.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D);a=np.array(a);print({'factor':'downside_adjusted_trend_60d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':np.mean(ns),'coverage':np.mean(cov),'ic':np.mean(a),'icir':np.mean(a)/np.std(a,ddof=1),'hit':np.mean(a>0),'turnover':np.mean(turn)})
for lab,m in [('365',D>=pd.Timestamp('2030-10-30')),('180',D>=pd.Timestamp('2031-04-30')),('60',D>=pd.Timestamp('2031-08-01'))]:
 z=a[m];print(lab,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1))
for h in [1,5,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in D:
  x=f.loc[d];y=yy.loc[d];ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_1_20311030_downside_trend_60d_signal.csv');pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_1_20311030_downside_trend_60d_ic.csv',index=False)
