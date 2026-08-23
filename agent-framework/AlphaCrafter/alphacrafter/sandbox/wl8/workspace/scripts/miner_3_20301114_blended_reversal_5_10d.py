import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2030-11-13')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();r=p.pct_change();rv=r.rolling(20).std()
# Blend short and medium volatility-scaled reversal; equal weights reduce horizon sensitivity.
f5=-p.pct_change(5)/(rv*np.sqrt(5)).clip(lower=.005); f10=-p.pct_change(10)/(rv*np.sqrt(10)).clip(lower=.005); f=(f5+f10)/2
D=[];A=[];N=[];C=[];T=[];prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 q=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(q):
  D.append(p.index[i]);A.append(q);N.append(int(ok.sum()));C.append(float(ok.mean()))
  if prev is not None:
   oo=x.notna()&prev.notna();T.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D);a=np.array(A)
print({'factor':'blended_vol_scaled_reversal_5_10d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':np.mean(N),'coverage':np.mean(C),'ic':a.mean(),'icir':a.mean()/a.std(ddof=1),'hit':(a>0).mean(),'turnover':np.mean(T)})
for n,m in [('180',D>=pd.Timestamp('2030-05-01')),('360',D>=pd.Timestamp('2029-11-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01')),('recent60',D>=pd.Timestamp('2030-08-01'))]:
 z=a[m];print(n,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_3_20301114_blended_reversal_5_10d_ic.csv',index=False)
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_3_20301114_blended_reversal_5_10d_signal.csv')
