import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-12-11')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index()
r=p.pct_change(); vol=r.rolling(20).std()
# Short-horizon contrarian signal, scaled by asset risk and market-wide dispersion regime.
base=-p.pct_change(5)/(vol*np.sqrt(5)).clip(lower=.005)
disp=r.rolling(20).std().mean(axis=1)
gate=(disp/disp.rolling(60).median()).clip(0.5,2.0)
f=base.mul(gate,axis=0)
D=[];A=[];N=[];C=[];T=[];prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 ic=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(ic):
  D.append(p.index[i]); A.append(ic); N.append(int(ok.sum())); C.append(float(ok.mean()))
  if prev is not None:
   oo=x.notna()&prev.notna(); T.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D); a=np.array(A)
print({'factor':'dispersion_reversal_10d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':np.mean(N),'coverage':np.mean(C),'ic':a.mean(),'icir':a.mean()/a.std(ddof=1),'hit':(a>0).mean(),'turnover':np.mean(T)})
for n,m in [('180',D>=pd.Timestamp('2030-06-01')),('360',D>=pd.Timestamp('2029-12-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01')),('recent60',D>=pd.Timestamp('2030-09-01'))]:
 z=a[m]; print(n,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_1_20301212_dispersion_reversal_10d_ic.csv',index=False)
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_1_20301212_dispersion_reversal_10d_signal.csv')
