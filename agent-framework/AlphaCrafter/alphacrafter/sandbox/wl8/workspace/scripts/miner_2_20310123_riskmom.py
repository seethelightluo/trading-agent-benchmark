import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2031-01-22')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();r=p.pct_change();f=p.pct_change(60)/r.rolling(20).std().clip(lower=.003)
D=[];A=[];N=[];C=[];T=[];prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-08-01') or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 q=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(q):
  D.append(p.index[i]);A.append(q);N.append(ok.sum());C.append(ok.mean())
  if prev is not None:
   oo=x.notna()&prev.notna();T.append(np.abs(x[oo].rank(pct=True)-prev[oo].rank(pct=True)).mean())
  prev=x
D=pd.DatetimeIndex(D);a=np.array(A);print({'factor':'risk_adjusted_momentum_60d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':round(np.mean(N),2),'coverage':round(np.mean(C),4),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1),6),'hit':round(np.mean(a>0),4),'turnover':round(np.mean(T),6)})
for lab,st in [('2029','2029-01-01'),('2030','2030-01-01'),('180','2030-07-01'),('90','2030-10-01'),('60','2030-11-01')]:
 z=a[D>=pd.Timestamp(st)];print(lab,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_2_20310123_riskmom_ic.csv',index=False);pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_2_20310123_riskmom_signal.csv')
