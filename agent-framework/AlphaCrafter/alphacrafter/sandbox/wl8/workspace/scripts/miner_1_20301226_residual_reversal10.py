import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-12-25')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=p.pct_change(); m=r.mean(axis=1); res=r.sub(m,axis=0); rv=r.rolling(20).std().clip(lower=.002); f=-res.rolling(10).sum()/(rv*np.sqrt(10))
D=[];A=[];N=[];C=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z):D.append(p.index[i]);A.append(z);N.append(ok.sum());C.append(ok.mean())
D=pd.DatetimeIndex(D);a=np.array(A);print({'factor':'residual_reversal_10d_volscaled_10d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':round(np.mean(N),2),'coverage':round(np.mean(C),4),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1),6),'hit':round((a>0).mean(),4)})
for q,mask in [('180',D>=pd.Timestamp('2030-06-01')),('90',D>=pd.Timestamp('2030-09-01')),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[mask];print(q,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_1_20301226_residual_reversal10_ic.csv',index=False);pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_1_20301226_residual_reversal10_signal.csv')
